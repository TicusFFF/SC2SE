# SC2SE：从自纠错到自进化

**From Self-Correction to Self-Evolution: Agentic Test-Time Learning for Mathematical Reasoning**  
Rui Zou · Yutao Zhu · Mengqi Wei · Ji-Rong Wen

SC2SE 把数学解题中的纠错经验整理成可复用的技能，供模型下一轮作答时使用。模型在测试时更新技能库，模型参数保持冻结，标准答案仅用于评分。

## 方法框架

[![SC2SE 的纠错、技能更新和技能应用三个阶段](assets/framework.png)](assets/framework.png)

求解智能体先作答，逐步纠错智能体检查解题过程，并根据发现的问题修正回答。技能进化智能体从这些记录中提炼解题方法，与已有技能合并。角色校准智能体在合并前后检查每条技能，确保它给出的是解题时可执行的建议。求解智能体带着更新后的技能重新作答，新一轮的验证和纠错又为技能更新提供经验。

## 主要结果

[![论文表 1：四个模型在五个数学题集上的准确率。末两行分别为使用技能的初答和进一步纠错后的回答。](assets/main-results.png)](assets/main-results.png)

表中 Base 是各方法共用的初答。SC2SE 从空技能库开始，在同一批题目上更新四轮；最后固定技能库，分别评价使用技能重新生成的初答（SC2SE<sub>init</sub>）和进一步纠错后的回答（SC2SE<sub>corr</sub>）。

## 安装与环境配置

在仓库根目录执行：

```bash
python -m pip install -r requirements.txt
```

设置 [`DEEPSEEK_API_KEY`](.env.example)。在 PowerShell 中使用 `$env:DEEPSEEK_API_KEY = "<your-key>"`，在 Bash 中使用 `export DEEPSEEK_API_KEY="<your-key>"`。示例通过 [DeepSeek 官方 API](https://api-docs.deepseek.com/) 调用 `deepseek-flash` 的非思考模式，并记录请求与返回的模型名。论文主表使用 DeepSeek V4.1-Flash、DeepSeek V4-Pro、Qwen3.6-35B-A3B 和 GPT-5.6-Luna。当前 Flash 接口别名不能保证与论文实验使用同一模型快照。

## 快速开始

[`demo`](0-0_datasets/hard_test_raw/demo/test_top2_filt2.json) 包含两道数学题。和其他题集一样，通过 `--dataset` 指定。

### 1. 生成公共初答

```bash
python 0-1_initial_response/generate_initial_responses.py --dataset demo --output-dir runs/initial
```

这一步也会评价初答，将结果写入 `runs/initial/demo.initial-eval.json`。已有匹配的初答和版本记录时，可将下一步的 `--initial-dir` 指向它所在的目录。

### 2. 运行 SC2SE 并评分

```bash
python 0-3_all_model_test/run_plan_S_SC/S_SC_S5-8-0_initialSolverSkill+strictACC+skillOptP1-1+compress/run_sc2se.py --dataset demo --initial-dir runs/initial --run
```

首轮从公共初答开始进行验证与纠错。两题示例也执行四轮技能更新，最后用固定的技能库作答并评分。运行目录中的 `demo/final_skill_evaluation/results.jsonl` 保存使用技能的初答和纠错后的回答，对应评分分别在 `evaluation_initial.json` 和 `evaluation_corrected.json`。Python 工具会在本机执行模型生成的代码，请使用专用、低权限的环境。

要改跑 HMMT，将两条命令中的 `--dataset demo` 换成 `--dataset hmmt_feb_2025`。其他题集的来源和准备方法见[数据说明](0-0_datasets/README.md)。

## 引用

Rui Zou, Yutao Zhu, Mengqi Wei, and Ji-Rong Wen. *From Self-Correction to Self-Evolution: Agentic Test-Time Learning for Mathematical Reasoning.*

[第三方代码、数据和许可证](THIRD_PARTY.md)
