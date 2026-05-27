# -*- coding: utf-8 -*-
"""Generate Word document: MetaBCI sleep project technical overview."""
import datetime
from docx import Document
from docx.shared import Inches, Pt, Cm, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT

doc = Document()
style = doc.styles['Normal']
style.font.name = 'Arial'
style.font.size = Pt(11)

# === Title ===
t = doc.add_heading('MetaBCI 睡眠分期项目 — 新增功能技术说明', level=0)
t.alignment = WD_ALIGN_PARAGRAPH.CENTER

p = doc.add_paragraph()
p.alignment = WD_ALIGN_PARAGRAPH.CENTER
r = p.add_run('晓途队 | 湘潭大学 | 2026 MetaBCI 创新应用开发赛 · 被动监测赛道')
r.bold = True
p.add_run(f'\n{datetime.date.today().strftime("%Y年%m月%d日")}')
doc.add_paragraph()

# === 1. Overview ===
doc.add_heading('一、项目概述', level=1)
doc.add_paragraph(
    '本项目基于 MetaBCI 开源框架，实现了一套完整的单通道（Fpz-Cz）便携式睡眠监测系统。'
    '系统以智能眼罩为硬件形态，通过 EEG 信号采集 → 在线推理 → 可视化报告的全链路，'
    '实现了对 W/N1/N2/N3/REM 五类睡眠阶段的实时监测。'
    '以下为在 MetaBCI 框架上新增的 7 个功能模块的详细技术说明。'
)

# === 2. Summary table ===
doc.add_heading('二、新增功能总览', level=1)

table = doc.add_table(rows=8, cols=5, style='Light Grid Accent 1')
table.alignment = WD_TABLE_ALIGNMENT.CENTER
headers = ['序号', '功能名称', '所属模块', '新增文件', '核心作用']
for i, h in enumerate(headers):
    cell = table.rows[0].cells[i]
    cell.text = h
    for pp in cell.paragraphs:
        for rr in pp.runs:
            rr.bold = True

data = [
    ['1', 'SleepEDFDataset\n数据集封装', 'brainda\ndatasets', 'sleep_edf.py',
     '加载Sleep-EDF数据库，自动发现153名受试者，解析PSG+Hypnogram配对文件'],
    ['2', 'SleepParadigm\n被动监测范式', 'brainda\nparadigms', 'sleep.py',
     '连续睡眠EEG处理范式，0.5-40Hz滤波+Z-score归一化钩子'],
    ['3', 'LWSleepNet\n轻量级模型', 'brainda\nalgorithms', 'lwsleepnet.py',
     '132K参数双分支深度可分离卷积+MHA模型，5分类睡眠分期'],
    ['4', 'SleepOnlineWorker\n在线推理器', 'brainflow', 'sleep_worker.py',
     '继承ProcessWorker基类，实时EEG流→滤波→归一化→推理→LSL推送'],
    ['5', 'SleepMonitorUI\n可视化界面', 'brainstim', 'sleep_monitor.py',
     '生成睡眠报告(Hypnogram+饼图+指标表)，实时+离线双模式'],
    ['6', 'ONNX导出+INT8量化', 'examples', 'export_onnx.py',
     'PyTorch→FP32 ONNX→INT8量化管线，模型缩小40%至347KB'],
    ['7', 'EDF回放模式', 'brainflow', 'edf_player.py',
     '模拟实时EEG数据流，支持倍速回放，联动Marker→Worker管线'],
]
for i, row_data in enumerate(data):
    for j, cell_text in enumerate(row_data):
        table.rows[i+1].cells[j].text = cell_text
doc.add_paragraph()

# === 3. Detailed ===
doc.add_heading('三、各功能详细技术说明', level=1)

features = [
    ('3.1 SleepEDFDataset', 'metabci/brainda/datasets/sleep_edf.py',
     [
         '继承BaseDataset，自动扫描目录（正则 r"SC(\\d{4})"）发现受试者',
         '_find_files() 自动配对 PSG-EDF 与 Hypnogram-EDF 文件',
         '_get_single_subject_data(): EDF加载→通道选择→0.3-45Hz滤波→100Hz重采样→解析Hypnogram→30s epoch切分→整数编码MNE注释',
         '关键兼容性：注释描述使用整数字符串，匹配BaseParadigm的 event_id=lambda x: int(x)',
         '阶段映射：W→0, N1→1, N2→2, N3/SWS→3, REM→4, ?/MT→-1（排除）',
         '用途：为模型训练和在线推理提供标准化数据输入。',
     ]),
    ('3.2 SleepParadigm', 'metabci/brainda/paradigms/sleep.py',
     [
         '继承BaseParadigm，paradigm="sleep"，与主动范式(MI/SSVEP/P300)区分',
         'is_valid(): 检查数据集的paradigm属性，确保范式-数据集匹配',
         'sleep_preprocess_hook: 0.5-40Hz带通滤波（作用于Raw阶段）',
         'sleep_normalize_hook: 逐epoch Z-score归一化（mean=0, std=1）',
         '事件配置：5类连续事件(W/N1/N2/N3/REM)，每类30秒窗口',
         '用途：提供睡眠分期专用数据处理流水线。',
     ]),
    ('3.3 LWSleepNet (核心创新)', 'metabci/brainda/algorithms/deep_learning/lwsleepnet.py',
     [
         '【架构】参考 Yang et al., Digital Health, 2023',
         '1. MultiResolutionBranch: k=5小核(高频∇20Hz+) + k=51大核(低频∇2Hz)，每分支 dw→pw→BN→GELU，拼接输64通道',
         '2. InvertedResidual1d: MobileNetV2风格倒残差，扩展×2→dw(k=9)→投影，残差连接',
         '3. TemporalAttentionBlock×3: 前置dw+pw→Patch切分(P=25)→8头MHA→序列重建→dw+pw→残差+BN',
         '4. 输出: AdaptiveAvgPool1d→Flatten→Dropout(0.5)→Linear(5)',
         '【指标】131,645参数，(B,1,3000)→(B,5)，支持FP32/FP64，@SkorchNet装饰',
         '用途：系统核心算法，极低参数量适合眼罩等边缘设备。',
     ]),
    ('3.4 SleepOnlineWorker', 'metabci/brainflow/sleep_worker.py',
     [
         '继承ProcessWorker(multiprocessing.Process)，完整接入brainflow实时管线',
         'pre(): 加载LWSleepNet模型到CPU，创建LSL StreamOutlet',
         'consume(data): EEG提取→补零/截断→4阶Butterworth(0.5-40Hz)→Z-score→(1,1,3000)→推理→argmax→LSL推送',
         'post(): 输出睡眠报告统计（各阶段时长、百分比）',
         '容错设计：变长输入自动补零、LSL不可用时降级、std+1e-8防除零',
         '用途：连接EEG设备(LSL/NeuroScan/Neuracle)实现实时睡眠推理。',
     ]),
    ('3.5 SleepMonitorUI', 'metabci/brainstim/sleep_monitor.py',
     [
         '模式1: build_sleep_report(predictions)—生成完整报告图(14×8英寸)',
         '  · Hypnogram—睡眠阶段时间线彩色填充图 (AASM标准颜色)',
         '  · 饼图—W/N1/N2/N3/REM分布百分比',
         '  · 指标表—TRT、睡眠潜伏期、TST、睡眠效率、深睡/REM占比',
         '模式2: SleepMonitorUI实时类—open()→update(stage)→close()，非阻塞matplotlib窗口',
         '模式3: generate_report(predictions, path)—一键生成PNG/PDF/SVG',
         '用途：展示整夜睡眠结构、阶段分布和关键质量指标。',
     ]),
    ('3.6 ONNX导出+INT8量化', 'examples/sleep_staging/export_onnx.py',
     [
         '完整管线: 加载.pth→FP32 ONNX(opset 17,动态batch)→INT8静态量化→精度验证→延迟基准',
         '量化配置: 500个真实EEG样本校准，weight和activation均量化为INT8',
         '实测结果: FP32=580KB, INT8=347KB(40%↓), PyTorch vs ONNX diff=0.000000',
         '用途：部署到眼罩MCU/嵌入式设备，INT8推理大幅降低计算和存储开销。',
     ]),
    ('3.7 EDF回放模式', 'metabci/brainflow/edf_player.py',
     [
         '组件1: EDFSleepPlayer(BaseAmplifier子类)—加载EDF→2提取通道→重采样→recv()返回[[ch, trigger],...]',
         'start_trans(): 1×/2×/∞倍速播放，自动匹配采样率节拍',
         '可选Hypnogram加载进行预测vs真实对照',
         '完全兼容 Marker→ProcessWorker 标准管线',
         '组件2: quick_test(model, edf_path)—离线快速测试，不经过brainflow管线',
         '用途：无硬件即可测试整套在线推理管线，支撑初赛视频演示。',
     ]),
]

for title, path, bullets in features:
    doc.add_heading(title, level=2)
    p = doc.add_paragraph(f'文件位置：{path}')
    p.runs[0].italic = True
    for b in bullets:
        doc.add_paragraph(b, style='List Bullet')
    doc.add_paragraph()

# === 4. Architecture ===
doc.add_heading('四、系统整体架构', level=1)
doc.add_paragraph(
    'Sleep-EDF → SleepEDFDataset → SleepParadigm → LWSleepNet (训练) → ONNX+INT8 (部署)\n'
    'EEG设备/EDF回放 → Marker(30s连续) → SleepOnlineWorker → SleepMonitorUI/LSL'
)

# === 5. Scoring ===
doc.add_heading('五、与竞赛评分标准的对应', level=1)

st = doc.add_table(rows=8, cols=3, style='Light Grid Accent 1')
st.alignment = WD_TABLE_ALIGNMENT.CENTER
for i, h in enumerate(['评分项', '分值', '本项目对应功能']):
    st.rows[0].cells[i].text = h
    for pp in st.rows[0].cells[i].paragraphs:
        for rr in pp.runs:
            rr.bold = True

sd = [
    ['brainda: 新增数据集', '10', '功能1: SleepEDFDataset'],
    ['brainda: 新增范式/算法', '15-20', '功能2: SleepParadigm + 功能3: LWSleepNet'],
    ['brainstim: 新增UI/可视化', '10', '功能5: SleepMonitorUI'],
    ['brainflow: 新增在线功能', '10', '功能4: SleepOnlineWorker + 功能7: EDFPlayer'],
    ['新增模型部署能力', '5-10', '功能6: ONNX+INT8量化'],
    ['改进与完善', '5', '单通道适配、连续Marker模式、容错设计'],
    ['100% MetaBCI完成度', '15', '全部7项基于MetaBCI三大模块开发'],
]
for i, row_data in enumerate(sd):
    for j, cell_text in enumerate(row_data):
        st.rows[i+1].cells[j].text = cell_text
doc.add_paragraph()

# === 6. Key metrics ===
doc.add_heading('六、关键技术指标', level=1)
metrics = [
    '模型参数量: 131,645 (~132K)，适合边缘部署',
    '输入规格: 单通道(Fpz-Cz)，30秒@100Hz = 3,000采样点',
    '输出类别: 5类(W/N1/N2/N3/REM)，符合AASM标准',
    '训练数据: Sleep-EDF Expanded, 153受试者, cassette子集',
    '模型格式: PyTorch(.pth) + FP32 ONNX(580KB) + INT8 ONNX(347KB)',
    '在线推理延迟: ~1ms/epoch (滤波+归一化+推理，不含数据采集)',
    'INT8体积缩减: 40% (580KB → 347KB)',
    '训练超参数: AdamW(β=0.9/0.999, wd=1), LabelSmoothing=0.05, Batch=120',
    '类不平衡策略: 平方根反比频率权重(sqrt inverse freq)，上限10×',
    'Python环境: 3.9.13, PyTorch 2.7.1+cu128, MNE 1.8.0',
]
for m in metrics:
    doc.add_paragraph(m, style='List Bullet')

# === 7. File list ===
doc.add_heading('七、新增文件清单', level=1)
files = [
    'metabci/brainda/datasets/sleep_edf.py              # SleepEDFDataset',
    'metabci/brainda/paradigms/sleep.py                 # SleepParadigm',
    'metabci/brainda/algorithms/deep_learning/',
    '    lwsleepnet.py                                  # LWSleepNet',
    'metabci/brainflow/sleep_worker.py                  # SleepOnlineWorker',
    'metabci/brainflow/edf_player.py                    # EDFSleepPlayer',
    'metabci/brainflow/__init__.py                      # 更新导出',
    'metabci/brainstim/sleep_monitor.py                 # SleepMonitorUI',
    'metabci/brainstim/__init__.py                      # 更新导出',
    'examples/sleep_staging/train_lwsleepnet.py         # 训练脚本',
    'examples/sleep_staging/export_onnx.py              # ONNX导出+量化',
    'verify_all.py                                      # 全模块验收',
]
for f in files:
    doc.add_paragraph(f, style='List Bullet')

# === 8. Dependencies ===
doc.add_heading('八、依赖清单', level=1)
deps = [
    ('Python', '3.9.13'),
    ('numpy', '2.0.2'), ('scipy', '1.13.1'), ('mne', '1.8.0'),
    ('torch', '2.7.1+cu128 (CUDA, RTX 5060)'), ('skorch', '1.2.0'),
    ('matplotlib', '(MNE自带)'), ('pylsl', '1.18.2'),
    ('onnx', '1.19.1 (新增)'), ('onnxruntime', '1.19.2 (新增)'),
]
dt = doc.add_table(rows=len(deps)+1, cols=2, style='Light Grid Accent 1')
dt.rows[0].cells[0].text = '依赖包'
dt.rows[0].cells[1].text = '版本'
for i, (n, v) in enumerate(deps):
    dt.rows[i+1].cells[0].text = n
    dt.rows[i+1].cells[1].text = v

# Save
output = r'C:\Users\lenovo\Desktop\MetaBCI-sleep\MetaBCI睡眠分期项目-新增功能技术说明.docx'
doc.save(output)
print(f'Done: {output}')
