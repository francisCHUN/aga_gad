# 完整闭环（推荐）
python -m aga.main --review-only --review-dir review_run --backend sohoyo --llm-precheck
#   → 打开 review_run/review_report.html，核对 LLM 预填建议，Export 两个文件放入 review_run/
python -m aga.main --resume-review review_run --backend sohoyo --cache-dir cache --train-compare --model h2fdetector

# refine 模块：全自动 gap 精炼循环（LLM-reviewer ↔ gap-refiner 多轮 refine，无暂停，CI 安全）
#   收敛或到达轮数预算后停止；精炼后的 gap set 不受约束地交给 Planner；审计包在 review_run/loop/
python -m aga.main --gap-loop --review-dir review_run --backend sohoyo --cache-dir cache --train-compare --model h2fdetector

# review + refine 闭环：精炼循环跑完后 PAUSE，人工终审精炼出的 gap set
python -m aga.main --review-gap-loop --review-dir review_run --backend sohoyo
#   → 打开 review_run/loop/review_report.html，curate 精炼后的 gaps
#     (confirmed=保留 / rejected=丢弃 / revised_confirmed=修改)，导出 loop/gap_feedback.json 放入 review_run/loop/
python -m aga.main --resume-gap-loop review_run --backend sohoyo --cache-dir cache --train-compare --model h2fdetector

# 旧的一次性流程（无审查，行为不变）
python -m aga.main --backend sohoyo --train-compare --model h2fdetector