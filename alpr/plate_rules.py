"""Quy tắc biển số Việt Nam: chuẩn hóa, sửa nhầm ký tự theo vị trí, kiểm tra định dạng (regex)."""
import re

# Mã tỉnh/thành trên biển số (danh sách theo mã biển số hiện hành, có thể bổ sung nếu có mã mới).
PROVINCE_CODES = {
    11, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28,
    29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 47, 48, 49,
    50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66,
    67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83,
    84, 85, 86, 88, 89, 90, 92, 93, 94, 95, 97, 98, 99,
}

# Ở vị trí phải là SỐ: chữ hay bị nhầm thành số
DIGIT_FIX = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1",
             "Z": "2", "S": "5", "G": "6", "T": "7", "B": "8"}
# Ở vị trí phải là CHỮ (seri): số hay bị nhầm thành chữ
LETTER_FIX = {"0": "D", "8": "B", "5": "S", "2": "Z", "6": "G", "4": "A"}

# 2 số mã tỉnh + 1-2 chữ seri + 4-6 số (6 số = có thêm 1 số ở seri, vd 59-C1 123.45)
PLATE_RE = re.compile(r"^(?P<prov>\d{2})(?P<letters>[A-Z]{1,2})(?P<digits>\d{4,6})$")
SERIES_DIGIT_FIX = {"I": "1", "L": "1", "T": "1", "Z": "2"}

def normalize(text: str) -> str:
    """Viết hoa, bỏ mọi ký tự không phải chữ/số (dấu '-', '.', khoảng trắng...)."""
    return re.sub(r"[^A-Z0-9]", "", text.upper().replace("Đ", "D"))


def fix_chars(s: str) -> str:
    """Sửa nhầm ký tự theo vị trí: 2 ký tự đầu là số, ký tự thứ 3 là chữ, cuối là số."""
    n = len(s)
    if not 7 <= n <= 9:
        return s
    num_len = 5 if n >= 8 else 4
    c = list(s)
    for i in (0, 1, *range(n - num_len, n)):
        c[i] = DIGIT_FIX.get(c[i], c[i])
    c[2] = LETTER_FIX.get(c[2], c[2])
    if n == 9:                                  # 2 số + 2 ký tự seri + 5 số -> thường là chữ + SỐ 
        c[3] = SERIES_DIGIT_FIX.get(c[3], c[3])
    return "".join(c)


def _build(fixed: str, m: re.Match, raw: str) -> dict:
    prov, letters, digits = m["prov"], m["letters"], m["digits"]
    sd, num = ("", digits)
    if len(digits) == 6:                       # biển xe máy dạng 59-C1 123.45
        sd, num = digits[0], digits[1:]
    num_fmt = f"{num[:3]}.{num[3:]}" if len(num) == 5 else num
    text = f"{prov}-{letters}{sd} {num_fmt}" if sd else f"{prov}{letters}-{num_fmt}"
    return {"raw": raw, "clean": fixed, "text": text, "valid": True,
            "province": int(prov), "province_ok": int(prov) in PROVINCE_CODES,
            "review": len(letters) == 2 or int(prov) not in PROVINCE_CODES}


def analyze(raw: str) -> dict:
    """Từ chuỗi OCR thô -> kết quả chuẩn hóa + cờ hợp lệ.

    valid = đúng cấu trúc biển số VN; province_ok = mã tỉnh nằm trong danh sách (chỉ là cảnh báo).
    review = hợp lệ nhưng đáng ngờ (seri 2 chữ, mã tỉnh lạ) -> nên thử lại/kiểm tra bằng mắt.
    """
    clean = normalize(raw)
    cands = [clean]
    # OCR có thể dư ký tự rác ở đầu/cuối -> thử thêm các cửa sổ con ngắn hơn (9, 8, 7 ký tự)
    cands += [clean[i:i + L] for L in (9, 8, 7) if L < len(clean) for i in range(len(clean) - L + 1)]
    for cand in cands:
        fixed = fix_chars(cand)
        m = PLATE_RE.match(fixed)
        if m:
            return _build(fixed, m, raw)
    return {"raw": raw, "clean": clean, "text": clean, "valid": False,
            "province": None, "province_ok": False, "review": True}


def merge_similar(counts: dict) -> dict:
    """Gộp các biển chỉ khác nhau <= 1 ký tự (lỗi OCR giữa các khung hình) về biển xuất hiện nhiều nhất.

    Lưu ý: hai xe có biển thật sự chỉ khác 1 ký tự sẽ bị gộp chung (hạn chế đã biết).
    """
    reps, mapping = [], {}
    for t, _ in sorted(counts.items(), key=lambda kv: -kv[1]):
        for r in reps:
            if len(r) == len(t) and sum(a != b for a, b in zip(r, t)) <= 1:
                mapping[t] = r
                break
        else:
            reps.append(t)
            mapping[t] = t
    return mapping