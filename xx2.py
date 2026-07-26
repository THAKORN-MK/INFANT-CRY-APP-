import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import os

path = 'C:/Windows/Fonts/Kanit-Light.ttf'
print("ไฟล์มีอยู่จริงไหม:", os.path.exists(path))

fm.fontManager.addfont(path)
prop = fm.FontProperties(fname=path)
print("ชื่อ font ที่ระบบอ่านได้:", prop.get_name())

fig, ax = plt.subplots()
ax.text(0.5, 0.5, 'ทดสอบภาษาไทย', fontproperties=prop, fontsize=20, ha='center')
fig.savefig('thai_font_test.png', dpi=100)
print("บันทึกรูปแล้ว")