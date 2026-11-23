"""Kiểm thử các phần không cần mô hình đã train.

    python -m unittest discover tests          # nhanh, không tải mô hình
    set LPR_TEST_OCR=1 && python -m unittest discover tests   # chạy cả PaddleOCR (~30 s, tải mô hình)
"""
import os
import tempfile
import unittest

import numpy as np

from lpr.metrics import iou, levenshtein, match_boxes
from lpr.plate_format import correct_plate, format_plate, is_valid
from lpr.plate_type import classify, plate_color, province
from lpr.tracker import IoUTracker
from tests.synthetic import make_plate


def fix(*lines):
    return format_plate(*correct_plate(list(lines)))


class TestPlateFormat(unittest.TestCase):
    def test_valid(self):
        for t in ("51F12345", "30A1234", "59X112345", "51LD12345", "29B11234"):
            self.assertTrue(is_valid(t), t)
        for t in ("", "51F", "F5112345", "51F1234567"):
            self.assertFalse(is_valid(t), t)

    def test_one_line(self):
        self.assertEqual(fix("51F12345"), "51F-123.45")
        self.assertEqual(fix("30A1234"), "30A-1234")
        self.assertEqual(fix("51LD12345"), "51LD-123.45")

    def test_confusion_fixed_by_position(self):
        self.assertEqual(fix("5lF1234S"), "51F-123.45")   # l->1, S->5
        self.assertEqual(fix("S1F-123.4S"), "51F-123.45")
        self.assertEqual(fix("518-123.45"), "51B-123.45")  # vị trí seri phải là chữ

    def test_two_lines(self):
        self.assertEqual(fix("59-X1", "123.45"), "59X1-123.45")
        self.assertEqual(fix("29B1", "1234"), "29B1-1234")
        self.assertEqual(fix("59X1", "I23.4S"), "59X1-123.45")


    def test_military(self):
        self.assertEqual(format_plate(*correct_plate(["TM-12-34"], military=True)), "TM-12-34")
        self.assertEqual(format_plate(*correct_plate(["8C-1Z-34"], military=True)), "BC-12-34")
        self.assertTrue(is_valid("TM1234", military=True))
        self.assertFalse(is_valid("TM1234"))  # không biết là biển đỏ -> không nhận dạng quân đội

    def test_military_when_color_missed(self):
        # Nền đỏ sẫm bị nhận màu "trắng": vẫn phải nhận là biển quân đội theo dạng chữ,
        # không được sửa kiểu dân sự BH5262 -> 8HS262.
        from lpr.ocr import build_result
        r = build_result(["BH-52-62"], 0.99, True, "trắng")
        self.assertEqual(r.display, "BH-52-62")
        self.assertTrue(r.valid)
        self.assertEqual(r.kind, "Biển quân đội")
        self.assertEqual(r.color, "đỏ")
        # Biển dân sự không bị ảnh hưởng
        self.assertEqual(build_result(["30E-931.05"], 0.99, True, "trắng").display, "30E-931.05")
        # Chuỗi rác từ biển mờ không được thành "biển quân đội": chữ đầu C không phải khối đơn vị,
        # hoặc đọc không rõ (tin cậy thấp) mà không có màu đỏ
        self.assertFalse(build_result(["CA2115"], 0.99, True, "trắng").valid)
        self.assertFalse(build_result(["BH5262"], 0.5, True, "trắng").valid)

    def test_same_line_grouping(self):
        # Gom vùng chữ PaddleOCR thành dòng (không cần tải mô hình: hàm tĩnh)
        try:
            from lpr.paddle_ocr import PaddlePlateOCR
        except ImportError:
            self.skipTest("chưa cài paddleocr")
        box = lambda x1, y1, x2, y2: np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], float)
        same = PaddlePlateOCR._same_line
        # Biển 2 dòng nghiêng: khung mỗi dòng cao, chồng ngang -> KHÔNG gộp (19A / 388.84)
        self.assertFalse(same(box(10, 30, 60, 75), [box(5, 0, 70, 50)]))
        # Mảnh vỡ cùng dòng nằm cạnh nhau -> gộp ("2" tách rời khỏi "99.68")
        self.assertTrue(same(box(0, 22, 8, 40), [box(10, 18, 60, 52)]))
        # 2 phần cùng dòng cùng độ cao (51F | 388.88) -> gộp
        self.assertTrue(same(box(70, 10, 160, 40), [box(0, 11, 60, 41)]))

    def test_series_letters(self):
        # Biển dân sự VN không dùng I, J, O, Q, W ở seri (vết bẩn cạnh "19A" bị đọc thành W)
        self.assertFalse(is_valid("19AW38884"))
        self.assertTrue(is_valid("19A38884"))
        for t in ("51F12345", "59X112345", "51LD12345", "41MD112345", "51R12345", "80NG12345"):
            self.assertTrue(is_valid(t), t)

    def test_diplomatic(self):
        self.assertEqual(fix("41-291-NN-01"), "41-291-NN-01")
        self.assertEqual(fix("41-636", "NN-90"), "41-636-NN-90")
        self.assertEqual(fix("4I-29I-NG-0I"), "41-291-NG-01")
        self.assertEqual(fix("80NG12345"), "80NG-123.45")
        self.assertTrue(is_valid("41291NN01"))

    def test_electric_motorbike(self):
        self.assertTrue(is_valid("41MD153904"))
        self.assertEqual(fix("41-MD1", "539.04"), "41MD1-539.04")
        self.assertEqual(fix("41-ND1", "539.04"), "41MD1-539.04")   # M đọc nhầm thành N
        self.assertEqual(fix("41MD153904"), "41MD1-539.04")
        self.assertIn("điện", classify("41MD1", "53904", 2))


class TestTrust(unittest.TestCase):
    """Chọn kết quả đáng tin hơn giữa các lần đọc (đọc thử nhiều cách trên cùng 1 biển)."""

    def test_prefers_letter_read_directly(self):
        from lpr.ocr import OCRResult, _trust
        # Lần đọc 1: M đọc thành 0 rồi bị đoán thành D; lần đọc 2 đọc thẳng ra M
        easy = OCRResult("51D99988", "51D-999.88", "51099988", 0.9, True)
        tess = OCRResult("51M99988", "51M-999.88", "51M999B8", 0.6, True)
        self.assertIs(max([easy, tess], key=_trust), tess)
        invalid = OCRResult("51M9998", "51M-9998", "51M9998", 0.99, False)
        self.assertIs(max([invalid, easy], key=_trust), easy)

    def test_one_line_fake_two_letter_series(self):
        from lpr.ocr import OCRResult, _trust
        # "30GC" (thừa chữ C) không phải seri thật của biển 1 dòng; "30G" đoán từ "306" thì có thể đúng
        extra = OCRResult("30GC03916", "30GC-039.16", "30GC03916", 0.85, True, ["30G-C03916"])
        guess = OCRResult("30G03916", "30G-039.16", "30603916", 0.85, True, ["306 039 16"])
        self.assertIs(max([extra, guess], key=_trust), guess)
        ld = OCRResult("51LD12345", "51LD-123.45", "51LD12345", 0.8, True, ["51LD12345"])
        self.assertTrue(_trust(ld)[0])                       # seri LD có thật
        moto = OCRResult("14AA17891", "14AA-178.91", "14AA17891", 0.8, True, ["14AA", "17891"])
        self.assertTrue(_trust(moto)[0])                     # xe máy 50cc 2 dòng


class TestRectify(unittest.TestCase):
    def test_tilted_plate_is_straightened(self):
        import cv2
        from lpr.rectify import rectify_plate
        p = make_plate(["59-X1", "123.45"], height=100)
        h, w = p.shape[:2]
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        dst = np.float32([[30, 20], [w + 20, 5], [w + 35, h + 10], [25, h + 30]])
        tilted = cv2.warpPerspective(p, cv2.getPerspectiveTransform(src, dst), (w + 70, h + 50),
                                     borderValue=(40, 40, 40))
        flat = rectify_plate(tilted)
        self.assertIsNotNone(flat)
        self.assertAlmostEqual(flat.shape[1] / flat.shape[0], w / h, delta=0.25)
        self.assertGreater(flat.mean(), tilted.mean())  # đã bỏ phần nền tối xung quanh

    def test_straight_plate_untouched(self):
        from lpr.rectify import rectify_plate
        self.assertIsNone(rectify_plate(make_plate(["51F-123.45"])))


class TestPlateType(unittest.TestCase):
    def test_classify(self):
        self.assertTrue(classify("51F", "12345").startswith("Ô tô"))
        self.assertTrue(classify("29B1", "1234", 2).startswith("Xe máy -"))
        self.assertTrue(classify("14AA", "17891", 2).startswith("Xe máy dưới 50"))
        self.assertIn("vận tải", classify("51F", "12345", 1, "vàng"))
        self.assertIn("nhà nước", classify("80A", "12345", 1, "xanh"))
        self.assertEqual(classify("TM", "1234", 1, "đỏ"), "Biển quân đội")
        self.assertIn("ngoại giao", classify("80NG", "12345"))
        self.assertIn("nước ngoài", classify("41291NN", "01"))
        self.assertIn("vốn nước ngoài", classify("51LD", "12345"))

    def test_province(self):
        self.assertEqual(province("51F12345"), "TP. Hồ Chí Minh")
        self.assertEqual(province("29B11234"), "Hà Nội")
        self.assertEqual(province("43A12345"), "Đà Nẵng")
        self.assertEqual(province("TM1234"), "")

    def test_color(self):
        for color in ("trắng", "vàng", "xanh", "đỏ"):
            self.assertEqual(plate_color(make_plate(["51F-123.45"], color=color)), color)

    def test_dark_text_inverted(self):
        from lpr.ocr import dark_text, preprocess_steps
        import cv2
        for color, dark in (("trắng", True), ("vàng", True), ("xanh", False), ("đỏ", False)):
            gray = cv2.cvtColor(make_plate(["51F-123.45"], color=color), cv2.COLOR_BGR2GRAY)
            self.assertEqual(dark_text(gray), dark, color)
            self.assertTrue(dark_text(preprocess_steps(gray)["Lọc bilateral"]), color)


class _FakeDetector:
    conf = 0.4

    def detect(self, img):
        return [(300, 10, 400, 40, 0.6), (10, 10, 110, 40, 0.9), (150, 10, 250, 40, 0.8)]


class _FakeOCR:
    def read(self, crop):
        from lpr.ocr import OCRResult
        return OCRResult("51F12345", "51F-123.45", "51F12345", 0.9, True, ["51F12345"], "trắng",
                         "Ô tô", "TP. Hồ Chí Minh")


class TestPipelineAndBatch(unittest.TestCase):
    def setUp(self):
        from lpr.pipeline import LPRPipeline
        self.pipe = LPRPipeline.from_parts(_FakeDetector(), _FakeOCR())
        self.img = np.full((100, 500, 3), 128, np.uint8)

    def test_multi(self):
        res = self.pipe.process_image(self.img, multi=True)
        self.assertEqual([r.box[0] for r in res], [10, 150, 300])  # trái -> phải
        one = self.pipe.process_image(self.img, multi=False)
        self.assertEqual(len(one), 1)
        self.assertEqual(one[0].det_conf, 0.9)

    def test_folder_and_excel(self):
        from openpyxl import load_workbook
        from lpr.batch import export_excel, list_images, process_paths, to_json
        from lpr.io import imwrite
        with tempfile.TemporaryDirectory() as d:
            imwrite(os.path.join(d, "a.jpg"), self.img)
            imwrite(os.path.join(d, "con", "b.png"), self.img)
            open(os.path.join(d, "ghi_chu.txt"), "w").close()
            paths = list_images(d)
            self.assertEqual([p.name for p in paths], ["a.jpg", "b.png"])
            self.assertEqual(len(list_images(d, recursive=False)), 1)
            recs = process_paths(self.pipe, paths, multi=True)
            self.assertEqual(recs[0].plate_list(), ["51F-123.45"] * 3)
            self.assertEqual(to_json(recs)[1]["so_bien"], 3)
            out = export_excel(recs, os.path.join(d, "kq.xlsx"))
            wb = load_workbook(out)
            self.assertEqual(wb.sheetnames, ["Biển số", "Theo ảnh", "Tổng hợp"])
            self.assertEqual(wb["Biển số"].max_row, 1 + 6)
            self.assertEqual(len(wb["Biển số"]._images), 6)
            self.assertEqual(wb["Theo ảnh"]["D2"].value, '["51F-123.45", "51F-123.45", "51F-123.45"]')
            wb.close()
            self.assertTrue(export_excel(recs, None).startswith(b"PK"))  # bytes cho web


class TestMetrics(unittest.TestCase):
    def test_iou(self):
        self.assertAlmostEqual(iou((0, 0, 10, 10), (0, 0, 10, 10)), 1.0)
        self.assertAlmostEqual(iou((0, 0, 10, 10), (5, 0, 15, 10)), 1 / 3)
        self.assertEqual(iou((0, 0, 1, 1), (2, 2, 3, 3)), 0.0)

    def test_levenshtein(self):
        self.assertEqual(levenshtein("51F12345", "51F12345"), 0)
        self.assertEqual(levenshtein("51F12345", "51F1234"), 1)
        self.assertEqual(levenshtein("51F12345", "81F12346"), 2)

    def test_match_boxes(self):
        gts = [(0, 0, 10, 10), (100, 100, 120, 110)]
        preds = [(101, 100, 121, 110), (50, 50, 60, 60), (1, 0, 10, 10)]
        self.assertEqual(match_boxes(preds, gts), {0: 2, 1: 0})


class TestTracker(unittest.TestCase):
    def test_keeps_id_while_moving(self):
        tr = IoUTracker()
        ids = [tr.update([(x, 0, x + 100, 30)])[0] for x in range(0, 50, 5)]
        self.assertEqual(len(set(ids)), 1)
        self.assertNotEqual(tr.update([(500, 500, 600, 530)])[0], ids[0])


@unittest.skipUnless(os.environ.get("LPR_TEST_OCR"), "đặt LPR_TEST_OCR=1 để chạy test PaddleOCR")
class TestPaddleOCR(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from lpr.paddle_ocr import make_fast
        cls.ocr = make_fast()

    def test_read(self):
        for lines, want in ((["51F-123.45"], "51F-123.45"), (["59-X1", "123.45"], "59X1-123.45")):
            r = self.ocr.read(make_plate(lines))
            self.assertEqual(r.display, want)
            self.assertTrue(r.valid)


if __name__ == "__main__":
    unittest.main()
