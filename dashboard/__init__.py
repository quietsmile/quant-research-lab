"""Web 仪表盘（Streamlit + Plotly）。

理念：UI 层薄、逻辑层厚。所有计算/绘图放在 core.py（可单测），
app.py 只负责把控件接到 core。未来要加指标监控，只在 core 加函数即可。
"""
