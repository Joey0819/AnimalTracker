"""
更新 app.py - 添加 v0.2.0 新功能
在侧边栏添加增强追踪器选项
"""

# 找到 app.py 中的追踪器选择部分并添加新选项
update_code = '''
    # 追踪算法选择
    st.markdown("---")
    st.subheader("追踪算法 (v0.2.0 新功能)")
    
    tracker_type = st.selectbox(
        "选择追踪器",
        ["基础追踪器", "改进追踪器（卡尔曼滤波）", "增强追踪器（v0.2.0）"],
        index=0
    )
    
    # 如果选择增强追踪器，显示更多选项
    if tracker_type == "增强追踪器（v0.2.0）":
        st.info("✨ 新功能：基于 STCS 论文")
        
        use_optical_flow = st.checkbox("启用光流估计（提高长期稳定性）", value=True)
        use_hungarian = st.checkbox("启用匈牙利匹配（提高准确率）", value=True)
        
        preset = st.selectbox(
            "预设配置",
            ["平衡模式（推荐）", "高精度模式", "快速模式"],
            index=0
        )
        
        if preset == "高精度模式":
            st.caption("适合：论文发表、关键实验")
        elif preset == "平衡模式":
            st.caption("适合：日常研究")
        else:
            st.caption("适合：实时处理")
    else:
        use_optical_flow = False
        use_hungarian = False
        preset = "balanced"
'''

print("app.py 更新代码已生成")
print("\n请在 app.py 中找到 '追踪算法选择' 部分")
print("替换为上面的代码")
print("\n或者在侧边栏添加以下内容：")
print(update_code)
