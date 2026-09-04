# 主题系统与 Design Tokens

平台视觉设计采用 **token 化主题系统**：颜色 / 圆角 / 间距 / 字号梯度统一
由 `app/models/style_tokens.py` 定义（当前唯一权威来源），新组件与后续迁移
从字面量逐步收敛到 token。

## 术语（Design Tokens）

| 类别 | 交代表 | 示例 |
|---|---|---|
| 颜色 | `COLORS` | `bg_panel` / `accent_blue` / `status_error` / `series_0..7` |
| 间距（8px 基线） | `SPACING` | `xs=4 sm=8 md=12 lg=16 xl=24 xxl=32` |
| 圆角 | `RADIUS` | `chip=10 card=14 panel=16 input=8` |
| 字号梯度 | `TYPOGRAPHY` | `micro=9 caption=10 body=12 section=15 page=17 display=22` |

```python
from app.models.style_tokens import color, series_color, SPACING, RADIUS

painter.setPen(QColor(color('accent_blue')))      # 新自绘组件
path.addRoundedRect(rect, RADIUS['card'], RADIUS['card'])
```

## 设计哲学落点

- **渐进式披露三层**：常驻（核心操作）→ 一键（次级操作折叠/芯片）→ 收纳
  （右键/高级区/对话框）。
- **有序专业密度**（Linear 式）：信息密度高但排版有序；不做消费级极简。
- **三硬规则**：画布 ≥55%（默认 splitter 18/62/20 + 右面板可折叠）、
  接近性分组（底部工具三分组/工具行分组）、上下文显隐替代禁用灰。
- **代码风格**：顶部任务芯片按任务着色、评估徽章 tone 化、Toast 按
  语义色（信息/成功/警告/错误）。

## 迁移路线（渐进，逐 PR 推进）

1. **新组件**：一律使用 token（自绘区、新面板）。
2. **QSS 迁移**：`resources/style.qss` 中高频字面量按 token 表替换
   （`#36B7FF → token accent_blue` 等）；建议以颜色为单位分步。
3. **多主题准备**：未来 Dark/Light 只需替换 `COLORS` 字典（UI 层
   不用改）。

## 主题令牌对照（Dark 现行）

| Token | 值 | 典型用途 |
|---|---|---|
| `bg_panel` | `#08101A` | 面板底色 |
| `bg_input` | `#0B141F` | 输入控件 |
| `border_mid` | `rgba(137,187,214,110)` | 卡片/分隔 |
| `accent_blue` | `#36B7FF` | 主强调 |
| `status_success` | `#45D483` | 成功/完成 |
| `status_error` | `#FF6B6B` | 错误/危险 |
| `status_warning` | `#F5A524` | 警告/待处理 |
| `series_0..7` | 八色系列 | 图表/图例 |
