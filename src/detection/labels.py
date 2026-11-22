"""Danh mục 13 nhãn yêu cầu; ID nghiệp vụ 1..13, chỉ số checkpoint bắt đầu từ 0. Bỏ Xe bán hàng rong."""

LABELS = {1: 'Người', 2: 'Xe máy', 3: 'Bàn', 4: 'Ghế', 5: 'Xe đẩy bán hàng', 6: 'Biển quảng cáo', 7: 'Cột điện', 8: 'Thân cây', 9: 'Hàng hóa bày bán', 10: 'Vật liệu xây dựng', 11: 'Máy móc', 12: 'Xô', 13: 'Phương tiện thu gom, vận chuyển rác'}

PHRASE_LABELS = {name: name for name in LABELS.values()}
