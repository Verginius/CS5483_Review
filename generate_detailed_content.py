import os
import json

m01 = """
<h1>MODULE 01: 概述与数据预处理</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>Overview.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>KDD流程全貌、数据特征类型的具体分类、缺失值与冗余特征处理方法、距离与相似度度量分析。</div>
<h2>1. 什么是数据挖掘与知识发现 (KDD)?</h2>
<p>数据挖掘 (Data Mining) 是知识发现 (Knowledge Discovery from Databases, KDD) 过程中最为核心的一环。完整的 KDD 包含以下 <strong>4 个核心步骤</strong>：</p>
<ol>
    <li><strong>Preparation (准备阶段)</strong>
        <ul>
            <li>设定挖掘目标（确定需要学习和发现的内容）。</li>
        </ul>
    </li>
    <li><strong>Pre-processing (数据预处理阶段)</strong>
        <ul>
            <li>数据清洗/数据集成 (Data cleaning/integration)：处理噪声、错误数据并填补缺失值。</li>
            <li>数据选择/归约/转换/离散化 (Data selection/reduction/transformation/discretization)：筛选相关样本与变量，构建用于挖掘的目标数据集。</li>
        </ul>
    </li>
    <li><strong>Data mining (数据挖掘阶段)</strong>
        <ul>
            <li>应用具体的机器学习算法，从预处理好的数据中提取出期望的模式 (patterns)。</li>
        </ul>
    </li>
    <li><strong>Interpretation (解释与部署阶段)</strong>
        <ul>
            <li>如果评估结果不理想，则返回前序步骤进行迭代 (Iterate)。</li>
            <li>如果结果符合预期，则执行部署 (Deploy)：包括生成报告、系统整合及落地应用。</li>
        </ul>
    </li>
</ol>
<h2>2. 数据集与属性的类型剖析</h2>
<p>在 WEKA 的 ARFF 文件格式中，属性类型的设定十分严格，算法的表现依赖特征的数据类型：</p>
<ul>
    <li><strong>Nominal / Categorical (标称型 / 类别型)</strong>：如天气（晴、阴、雨）。在 WEKA 中表示为 <code>{Sunny, Overcast, Rainy}</code>。此类数据无大小顺序，不能进行代数运算。多数关联规则算法（如 Apriori）要求数据必须为标称型。</li>
    <li><strong>Ordinal (序数型)</strong>：如满意度（高、中、低）。具有明确顺序，但间隔无数学意义。在算法处理时可转化为数值或标称形式。</li>
    <li><strong>Numeric / Continuous (数值型 / 连续型)</strong>：包含区间型 (Interval) 和比率型 (Ratio，有绝对零点)。可计算均值和方差，是 KNN 等算法计算欧氏距离的基础类型。在 WEKA 中用 <code>numeric</code> 表示。</li>
</ul>

<h2>3. WEKA 数据预处理核心算法流程</h2>
<p><strong>ReplaceMissingValues 过滤器执行流程:</strong></p>
<p><strong>输入：</strong>包含缺失值的数据集。<br>
<strong>输出：</strong>填补了缺失值的完整数据集。</p>
<ol>
    <li><strong>[扫描特征]</strong>：扫描数据集中的每一列特征属性。</li>
    <li><strong>[处理数值型]</strong>：如果该列为 <strong>Numeric (数值型)</strong>，计算该列所有非缺失样本的 <strong>均值 (Mean)</strong>，将所有缺失的 <code>?</code> 替换为均值。</li>
    <li><strong>[处理标称型]</strong>：如果该列为 <strong>Nominal (标称型)</strong>，统计该列各种分类的出现频次，找出 <strong>众数 (Mode)</strong>，将所有缺失的 <code>?</code> 替换为众数。</li>
    <li><strong>[利用类标签优化]</strong>：如果数据集包含类别标签 (Class Label)，该过滤器可选择是否针对每个类别的内部独立计算均值/众数，从而提高填补的精确度。</li>
</ol>

<h2>4. 距离与相似度度量详解</h2>
<p>数据点之间的接近程度是基于实例学习（如 KNN、K-Means 聚类）的核心：</p>
<ul>
    <li><strong>Minkowski Distance (闵可夫斯基距离)</strong>：广义距离公式 $L_p$ 范数。<br>
        - 当 $p=1$ 时，退化为 <strong>Manhattan (曼哈顿距离)</strong>，计算绝对轴距和：$d(x, y) = \\sum_{i=1}^n |x_i - y_i|$。<br>
        - 当 $p=2$ 时，退化为 <strong>Euclidean (欧氏距离)</strong>，最常用于连续型特征，但对大数值敏感：$d(x, y) = \\sqrt{\\sum_{i=1}^n (x_i - y_i)^2}$。<br>
        - 当 $p \\rightarrow \\infty$ 时，为 <strong>Supremum / Chebyshev 距离</strong>（切比雪夫距离）：$d(x, y) = \\max_i |x_i - y_i|$。
    </li>
    <li><strong>Cosine Similarity (余弦相似度)</strong>：常用于高维稀疏数据（如文本向量），测量向量夹角，忽略数值的绝对大小，关注方向的匹配度：$\\cos(\\theta) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}$。</li>
    <li><strong>Jaccard Coefficient (雅卡尔系数)</strong>：适用于非对称的二元布尔属性（例如顾客是否购买了某件商品），忽略双 0 匹配，公式为 $J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}$。</li>
</ul>
<h3>🎯 经典例题</h3>
<p><strong>【Q1. 简答题】</strong>请列举出完整的 KDD (Knowledge Discovery from Databases) 过程所包含的 4 个核心步骤，并指明哪一个大步骤包含了 Deploy (部署)？</p>
<p><strong>【解答】</strong>: 完整的 KDD 步骤包含 4 大阶段：<br>1. Preparation (准备)<br>2. Pre-processing (预处理)<br>3. Data mining (数据挖掘)<br>4. Interpretation (解释与部署)<br>其中，部署 (Deploy) 属于第四个核心阶段 <strong>Interpretation (解释)</strong>。</p>
<p><strong>【Q2. 计算题】</strong>给出两个向量 $X = (1, 3)$ 和 $Y = (4, 7)$，请分别计算它们之间的曼哈顿距离、欧氏距离和切比雪夫距离。</p>
<p><strong>【解答】</strong>:<br>
- 曼哈顿距离 ($L_1$): $|1-4| + |3-7| = 3 + 4 = 7$<br>
- 欧氏距离 ($L_2$): $\\sqrt{(1-4)^2 + (3-7)^2} = \\sqrt{9+16} = 5$<br>
- 切比雪夫距离 ($L_\\infty$): $\\max(|1-4|, |3-7|) = \\max(3, 4) = 4$</p>
"""

m02 = """
<h1>MODULE 02: 分类模型性能评估</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>Classification_EV.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>混淆矩阵推导、Accuracy在数据不平衡下的局限性、Recall与Precision的取舍、F-score、ROC曲线与PRC曲线对比。</div>
<h2>1. 混淆矩阵 (Confusion Matrix for Binary Classification)</h2>
<p>评估模型的直观工具。在 WEKA 输出结果中，行通常代表真实类别，列代表预测类别。</p>
<ul>
    <li><strong>TP (真正例)</strong>: 正类被正确预测为正类。</li>
    <li><strong>TN (真负例)</strong>: 负类被正确预测为负类。</li>
    <li><strong>FP (假正例 / 误报 False Alarm)</strong>: 负类被错误预测为正类。在统计学中称为 <strong>第一类错误 (Type I Error)</strong>。</li>
    <li><strong>FN (假负例 / 漏报 Missed Detection)</strong>: 正类被错误预测为负类。在统计学中称为 <strong>第二类错误 (Type II Error)</strong>。</li>
</ul>
<h2>2. 评价指标及其适用场景</h2>
<ul>
    <li><strong>Accuracy (准确率)</strong> = $\\frac{TP + TN}{TP + TN + FP + FN}$。<strong>局限性</strong>：在处理 <strong>Class imbalance problem (类别不平衡问题)</strong> 时参考价值低。例如当多数类占 99.9% 时，即使模型将所有样本均预测为多数类，准确率也可达 99.9%，但无法识别少数类。</li>
    <li><strong>Precision (精确率 / 查准率)</strong> = $\\frac{TP}{TP + FP}$。表示预测为正类的样本中，真实为正类的比例。适用于误报成本较高的场景。</li>
    <li><strong>Recall / Sensitivity / TPR (召回率 / 查全率 / 真正例率)</strong> = $\\frac{TP}{TP + FN}$。表示所有真实正类样本中，被正确预测出的比例。<strong>在疾病检测、风险预警中更为重要</strong>，因为漏报成本远高于误报成本。</li>
    <li><strong>Specificity / TNR (特异性 / 真负例率)</strong> = $\\frac{TN}{TN + FP}$。表示所有真实负类样本中，被正确预测出的比例。</li>
</ul>
<h2>3. 综合评价指标 (F-Measure)</h2>
<p>Precision 和 Recall 通常呈负相关关系。为了进行综合评估，常采用它们的调和平均数：<br>
<strong>F1-score</strong> = $2 \\times \\frac{Precision \\times Recall}{Precision + Recall}$。<br>
F1-score 仅在 Precision 和 Recall 均较高时才会取得高值。</p>
<p><strong>F-beta score</strong> 允许在两者间设定权重：$F_\\beta = (1 + \\beta^2) \\times \\frac{Precision \\times Recall}{\\beta^2 \\times Precision + Recall}$。当 $\\beta > 1$ 时更看重 Recall；当 $\\beta < 1$ 时更看重 Precision。</p>

<h2>4. WEKA 中的基础判别与阈值算法流</h2>
<p><strong>ZeroR 算法执行流程 (基准分类器):</strong></p>
<p><strong>输入：</strong>带有目标类别标签的训练数据集；测试样本。<br>
<strong>输出：</strong>对所有测试样本的多数类预测结果。</p>
<ol>
    <li><strong>[扫描全库]</strong>：忽略所有特征变量，仅统计目标类别 (Class Label) 的分布。</li>
    <li><strong>[建立模型]</strong>：强制预测为样本中数量最多的类别（多数类）。</li>
    <li><strong>[测试阶段]</strong>：对于任何新样本，均预测为多数类。常用于作为模型性能下限基准。</li>
</ol>
<p><strong>阈值移动与曲线绘制流程 (ROC/PRC 绘制原理):</strong></p>
<p><strong>输入：</strong>测试样本及其真实类标；模型输出的正类概率得分。<br>
<strong>输出：</strong>ROC 曲线或 PRC 曲线。</p>
<ol>
    <li><strong>[输出概率]</strong>：模型为每个测试样本输出其属于正类的 <strong>概率得分 (Probabilities)</strong>。</li>
    <li><strong>[排序]</strong>：将所有样本按照概率得分从高到低排序。</li>
    <li><strong>[初始阈值]</strong>：初始时，将判定阈值设为无穷大，所有样本被预测为负例（TP=0, FP=0）。</li>
    <li><strong>[降低阈值]</strong>：逐步降低阈值，每次将一个新样本判定为正例。如果其真实标签为正，则 TPR (Recall) 增加；如果为负，则 FPR 增加。</li>
    <li><strong>[绘制曲线]</strong>：当阈值降至0时，所有样本均被判定为正例，形成从 (0,0) 到 (1,1) 的评估曲线。</li>
</ol>

<h2>5. ROC 曲线与 PRC 曲线对比</h2>
<ul>
    <li><strong>ROC Curve (接收者操作特征曲线)</strong>：纵轴为 TPR (Recall)，横轴为 FPR ($1 - Specificity = \\frac{FP}{FP + TN}$)。理想模型的曲线靠近左上角 $(0, 1)$。<strong>AUC (Area Under Curve)</strong> 表示随机抽取一个正例得分高于负例的概率，数值越接近 1 越好。</li>
    <li><strong>PRC Curve (Precision-Recall 曲线)</strong>：纵轴为 Precision，横轴为 Recall。</li>
    <li><strong>使用场景对比</strong>：在处理极端类别不平衡数据（负例数量庞大）时，FP 的增加相对于庞大的 TN 显得极小，导致 FPR 变化不明显，ROC 曲线会显得过于乐观。而 Precision 的分母为 $TP+FP$，排除了 TN 的影响，能更真实地反映模型性能。因此，<strong>面对严重不平衡数据，应首选 PRC 进行评估。</strong></li>
</ul>
<h3>🎯 经典例题</h3>
<p><strong>【Q1. 矩阵计算题】</strong>在一次针对罕见病的预测中，WEKA 输出了以下混淆矩阵 (行代表实际，列代表预测)：<br>
<pre>
   a   b  <-- classified as
 100  20  | a = 患病 (Positive)
  30 850  | b = 健康 (Negative)
</pre>
请计算 患病类别(Positive) 的 Accuracy, Precision, Recall 和 F1-score。</p>
<p><strong>【解答】</strong>:<br>
- 基础值：$TP=100, FN=20, FP=30, TN=850$<br>
- $Accuracy = \\frac{100+850}{100+20+30+850} = 95\\%$<br>
- $Precision = \\frac{100}{100+30} \\approx 76.92\\%$<br>
- $Recall = \\frac{100}{100+20} \\approx 83.33\\%$<br>
- $F1\\text{-}score = 2 \\times \\frac{0.7692 \\times 0.8333}{0.7692 + 0.8333} \\approx 80\\%$</p>
<p><strong>【Q2. 概念辨析】</strong>在极端不平衡的信用卡欺诈检测数据集中，为什么即使模型性能不佳，ROC 曲线下的面积 (AUC) 也可能较高？请指出一种更合适的替代评估图表。</p>
<p><strong>【解答】</strong>: 在极端不平衡数据中，真实负例 (TN) 的数量远大于假正例 (FP)。ROC 的横轴 $FPR = FP / (FP+TN)$，由于 $TN$ 极大，即使模型出现较多假阳性错误，$FPR$ 依然保持在较低水平，使 ROC 曲线表现出过于乐观的形态。<br>此时应使用 <strong>Precision-Recall Curve (PRC)</strong>，由于 Precision 的分母是 $TP+FP$，不受 $TN$ 影响，能更准确地反映模型在少数类上的预测能力。</p>
"""

m03 = """
<h1>MODULE 03: Decision Tree Induction (决策树)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>Classification_DT.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>贪心算法的特性、Information Gain 的多值偏向问题及其修正 (Gain Ratio)、CART树的 Gini 纯度、连续与离散变量处理、WEKA中决策树算法 (J48, REPTree) 的比较与剪枝机制。</div>
<h2>1. 算法核心原理：贪心算法 (Greedy Algorithm)</h2>
<p>决策树采用自上而下的 <strong>Divide and Conquer (分而治之)</strong> 策略，在每一步分裂中寻找使子集纯度提升最大的属性。<br>
<strong>局限性 (Myopic Problem)</strong>：由于采用贪心策略，决策树每次分裂仅关注当前步骤的局部最优解，不考虑后续分裂的组合效应，可能无法达到全局最优状态。</p>
<h2>2. 杂质度量与分裂准则 (Impurity Measures)</h2>
<p>决策树的主要分裂标准包括：</p>
<ul>
    <li><strong>Shannon's Entropy (信息熵)</strong>：计算公式 $H(p) = -\\sum p_i \\log_2(p_i)$，反映数据的混乱程度。类别分布越均匀，熵越高。
        <ul>
            <li><strong>Information Gain (信息增益, ID3使用)</strong> = 父节点熵 - 子节点加权熵。倾向于选择能最大程度降低混乱度的属性。<br>
            <strong>偏向性缺陷</strong>：若某属性具有大量唯一值（如“ID”），其分裂将产生大量纯度极高的单一分支，信息增益会达到最大。但此种分裂缺乏泛化能力，容易导致过拟合。</li>
            <li><strong>Gain Ratio (增益率, C4.5使用)</strong> = $\\frac{Information\\ Gain}{Split\\ Information}$。Split Info 为属性本身分裂产生的熵：$SplitInfo_A(D) = -\\sum_{j=1}^v \\frac{|D_j|}{|D|} \\log_2 \\left(\\frac{|D_j|}{|D|}\\right)$。多值属性的 Split Info 较大，作为分母有效抑制了对多值属性的偏向。</li>
        </ul>
    </li>
    <li><strong>Gini Impurity (基尼不纯度, CART使用)</strong>：公式 $Gini(p) = 1 - \\sum_{i=1}^J p_i^2$。衡量随机抽取两个样本其类别不一致的概率。Gini 在二元分类下计算效率较高，通常倾向于划分大小相对均衡的子集。</li>
</ul>

<h2>3. WEKA 决策树算法：J48, REPTree 与 RandomTree 流程</h2>
<p><strong>J48 算法 (对应 C4.5) 核心建树流程:</strong></p>
<p><strong>输入：</strong>包含特征属性与类标的训练数据集。<br>
<strong>输出：</strong>一棵修剪过的决策树模型。</p>
<ol>
    <li><strong>[同类判定]</strong>：若当前节点内所有样本 <strong>属于同一类别</strong>，则设为叶子节点。</li>
    <li><strong>[结束判定]</strong>：若 <strong>无剩余属性可供划分</strong> 或样本数少于 <code>minNumObj</code> 下限，按多数表决法设为叶子节点。</li>
    <li><strong>[计算增益]</strong>：计算所有候选属性的 <strong>Gain Ratio (增益率)</strong>。</li>
    <li><strong>[选择属性]</strong>：选择带来 <strong>最大纯度提升 (Best Split)</strong> 的属性作为分裂节点。</li>
    <li><strong>[递归划分]</strong>：根据该属性将数据集划分为 <strong>子集 (Subsets)</strong>，对每个子集 <strong>递归</strong> 调用步骤1到4。</li>
    <li><strong>[后剪枝]</strong>：树构建完成后，执行 <strong>后剪枝 (Post-pruning)</strong>。利用参数 <code>confidenceFactor</code> 评估剪枝前后的误差成本，进行修剪以防过拟合。</li>
</ol>

<p><strong>REPTree (Reduced Error Pruning Tree) 算法核心流程:</strong></p>
<p><strong>输入：</strong>带有特征与类标的原始数据集。<br>
<strong>输出：</strong>一棵经过 Reduced Error Pruning 优化的决策树。</p>
<ol>
    <li><strong>[数据划分]</strong>：将数据集划分为 <strong>训练集</strong> 和 <strong>独立验证集 (Holdout set)</strong>。</li>
    <li><strong>[快速建树]</strong>：使用训练集，基于 Information Gain 或方差快速构建决策树。</li>
    <li><strong>[泛化评估]</strong>：自底向上利用 <strong>独立验证集</strong> 评估各分支的泛化能力。</li>
    <li><strong>[实施剪枝]</strong>：若某分支节点在验证集上的误差高于将其替换为单一叶节点的误差，则执行 <strong>剪枝 (Reduced Error Pruning)</strong>。</li>
</ol>

<p><strong>RandomTree 算法核心流程:</strong></p>
<p><strong>输入：</strong>训练数据集；特征子集大小 $K$。<br>
<strong>输出：</strong>一棵未剪枝的随机特征决策树。</p>
<ol>
    <li><strong>[屏蔽全量特征]</strong>：在计算分裂节点时，不考虑所有可用特征。</li>
    <li><strong>[随机抽样特征]</strong>：从所有 $M$ 个特征中，<strong>随机抽取 $K$ 个特征组成子集</strong>（通常 $K=\\log_2M+1$）。</li>
    <li><strong>[特征选择]</strong>：仅在这 $K$ 个特征中选择最优特征进行分裂。</li>
    <li><strong>[完全生长]</strong>：建树过程 <strong>不执行剪枝 (No pruning)</strong>，允许树充分生长。</li>
    <li><strong>[集成准备]</strong>：由于单棵树方差较大，通常将其作为 RandomForest (随机森林) 的基础分类器，通过集成平滑误差。</li>
</ol>

<h3>🎯 经典例题</h3>
<p><strong>【Q1. 计算推演题】</strong>在一个包含 10 条训练样本的节点中，有 6 个是 "Yes"，4 个是 "No"。利用属性 <code>Wind</code> (具有两个分支：Strong 和 Weak) 对其分裂，其中 Strong 分支包含 2 个 "Yes" 和 3 个 "No"；Weak 分支包含 4 个 "Yes" 和 1 个 "No"。请写出该次分裂 Information Gain 的计算公式。</p>
<p><strong>【解答】</strong>:<br>
- 父节点的初始信息熵：$H(S) = -\\left(\\frac{6}{10} \\log_2 \\frac{6}{10} + \\frac{4}{10} \\log_2 \\frac{4}{10}\\right)$<br>
- Strong 分支（共5条）的局部熵：$H(Strong) = -\\left(\\frac{2}{5} \\log_2 \\frac{2}{5} + \\frac{3}{5} \\log_2 \\frac{3}{5}\\right)$<br>
- Weak 分支（共5条）的局部熵：$H(Weak) = -\\left(\\frac{4}{5} \\log_2 \\frac{4}{5} + \\frac{1}{5} \\log_2 \\frac{1}{5}\\right)$<br>
- 最终的 Information Gain：$IG = H(S) - \\left( \\frac{5}{10} H(Strong) + \\frac{5}{10} H(Weak) \\right)$</p>
<p><strong>【Q2. 算法填空题】</strong>在 WEKA 中，如果希望建立一棵极简的决策树，要求叶子节点的样本不少于 20 个，且执行严格剪枝，应该使用 ____ 算法，将参数 <code>minNumObj</code> 设为 ____，并将参数 <code>confidenceFactor</code> 设得更 ____ (大/小)？</p>
<p><strong>【解答】</strong>: 应该使用 <code>J48</code> 算法，将参数 <code>minNumObj</code> 设为 <code>20</code>，并将参数 <code>confidenceFactor</code> 设得更 <code>小</code>（较小的 $C$ 值代表对误差波动的容忍度更低，从而触发更多的剪枝）。</p>
"""

m04 = """
<h1>MODULE 04: K-Nearest Neighbor (KNN)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>Classification_NN.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>懒惰学习 (Lazy Learning) 特性、Voronoi 图决策边界、距离度量细节、未归一化对距离计算的影响、以及 WEKA IBk 算法的原理。</div>
<h2>1. 懒惰学习 (Lazy Learning) 与 急切学习 (Eager Learning)</h2>
<p>多数算法（如决策树、神经网络）属于 <strong>Eager Learning</strong>，在训练阶段构建抽象模型，抛弃原始训练数据，预测阶段耗时较短。</p>
<p>KNN 是一种典型的 <strong>Lazy Learner (懒惰学习器)</strong>。它在训练阶段几乎没有计算开销，仅将训练样本保存于内存中。所有的计算被推迟至 <strong>预测阶段 (Testing phase)</strong>，通过计算测试样本与所有历史样本的距离来做出决策。</p>
<h2>2. Voronoi Diagram 与决策边界</h2>
<p>KNN 不生成显式的数学函数边界。当设定 $K=1$ 时，样本空间被划分为多个多边形区域，每个区域被一个训练样本主导，落入该区域的新样本距离该主导样本最近。此类由距离划分的几何网格被称为 <strong>Voronoi Diagram (沃罗诺伊图)</strong>。</p>
<h2>3. 特征量纲与归一化 (Normalization) 的重要性</h2>
<p>在使用欧氏距离 $D = \\sqrt{\\sum (x_i - y_i)^2}$ 计算时，若特征量纲存在巨大差异，<strong>数值范围大的特征将主导距离的计算结果，导致数值较小的特征失效</strong>。</p>
<p><strong>常用数据标准化方案：</strong></p>
<ul>
    <li><strong>Min-Max Normalization (最小-最大规范化)</strong>：将数据线性映射至 $[0, 1]$ 区间。公式：$v' = \\frac{v - \\min}{\\max - \\min}$。此方法对离群点 (outliers) 较为敏感。</li>
    <li><strong>Z-score Standardization (标准正态规范化)</strong>：公式 $v' = \\frac{v - \\mu}{\\sigma}$，基于均值和标准差进行转换，处理离群点时稳定性更佳。</li>
</ul>
<h2>4. 超参数 $K$ 值的影响</h2>
<ul>
    <li><strong>$K$ 过小 (如 $K=1$)</strong>：决策边界复杂且支离破碎，模型容易受局部噪声影响，导致 <strong>过拟合 (Overfitting)</strong>。</li>
    <li><strong>$K$ 过大 (如 $K=N$)</strong>：模型失去区分能力，将所有样本预测为多数类 (Majority Class)，导致 <strong>欠拟合 (Underfitting)</strong>。</li>
</ul>

<h2>5. WEKA 中的 KNN 算法实现：IBk</h2>
<p>在 WEKA 中，KNN 的对应算法为 <strong>IBk</strong> (Instance-Based learning)。其工作流程如下：</p>
<p><strong>输入：</strong>带有类标的训练数据集；新测试实例 $X$；邻居数量 $K$。<br>
<strong>输出：</strong>新实例 $X$ 的预测类别。</p>
<ol>
    <li><strong>[Training 阶段]</strong>：将带有标签的训练数据保存至数据结构（如 <code>LinearNNSearch</code> 或 <code>KDTree</code>）中，不建立任何概率模型。</li>
    <li><strong>[属性归一化]</strong>：进入 <strong>Testing 阶段</strong>，对于新实例 $X$，IBk 可根据设置对其属性进行归一化，使其与历史数据的刻度一致。</li>
    <li><strong>[计算距离]</strong>：在空间中计算 $X$ 与内存中所有点的几何距离。</li>
    <li><strong>[选取邻居]</strong>：选出距离 $X$ 最近的 $K$ 个历史邻居。</li>
    <li><strong>[权重裁决]</strong>：若不加权，则前 $K$ 个邻居进行等权重投票；若启用 <strong>Inverse Distance (距离倒数加权)</strong>，则根据 $w_i = \\frac{1}{d(x, x_i)}$ 赋予距离更近的邻居更高的投票权重，以降低远处噪点的影响。最终由得票最高者决定 $X$ 的类别。</li>
</ol>

<h3>🎯 经典例题</h3>
<p><strong>【Q1. 归一化与距离计算题】</strong>有三个样本拥有两个特征(Salary, Age)：$A(50000, 30)$，$B(60000, 40)$，新样本 $X(55000, 35)$。已知 Salary 的极小极大值是 $[10000, 110000]$，Age 的极小极大值是 $[20, 70]$。请使用 Min-Max 归一化处理数据，并计算新样本 $X$ 距离 $A$ 和 $B$ 的欧氏距离。</p>
<p><strong>【解答】</strong>:<br>
- $Salary$ 归一化分母 = $110000-10000 = 100000$。$A_{sal}'=0.4, B_{sal}'=0.5, X_{sal}'=0.45$。<br>
- $Age$ 归一化分母 = $70-20 = 50$。$A_{age}'=0.2, B_{age}'=0.4, X_{age}'=0.3$。<br>
- 归一化后坐标：$A(0.4, 0.2)$， $B(0.5, 0.4)$， $X(0.45, 0.3)$。<br>
- 距离 $X$ 到 $A$: $\\sqrt{(0.45-0.4)^2 + (0.3-0.2)^2} = 0.1118$<br>
- 距离 $X$ 到 $B$: $\\sqrt{(0.45-0.5)^2 + (0.3-0.4)^2} = 0.1118$。<br>所以 $X$ 到 $A$ 和 $B$ 的距离相等。</p>
<p><strong>【Q2. 概念辨析】</strong>如果 KNN 模型发生了严重的过拟合 (Overfitting)，即在测试集上表现极差，我们通常应该增大还是减小 $K$ 值？为什么？</p>
<p><strong>【解答】</strong>: 应该 <strong>增大 $K$ 值</strong>。过拟合说明模型过度关注局部噪声和极端离群点。增大 $K$ 值可以让更多的邻居参与投票，从而平滑个别噪点的影响，使决策边界更加稳定。</p>
"""

m05 = """
<h1>MODULE 05: Rule-Based Classification (基于规则的分类)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>Classification_RB.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>规则分类的解释性优势、Sequential Covering (分离并征服) 策略、RIPPER 的 MDL 停止准则、WEKA中 JRip、PART 算法的运作机制。</div>
<h2>1. 基于规则分类的特性</h2>
<p>Rule-Based 系统在金融风控、医疗等领域应用广泛，其核心优势在于 <strong>IF-THEN 规则</strong> 具备直接阅读、审计和局部修改的特性（<strong>模块化</strong> 与 <strong>高解释性</strong>）。与决策树牵一发而动全身的结构不同，规则集可以直接禁用或调整单条不合理的规则。</p>
<h2>2. 规则的覆盖冲突 (Conflict Resolution)</h2>
<p>当测试实例同时触发多条互相矛盾的规则时，通常采用以下策略解决冲突：</p>
<ul>
    <li><strong>Size ordering</strong>：优先采用触发条件更苛刻、更具体的规则。</li>
    <li><strong>Class-based ordering</strong>：按类别的罕见性或重要性排序。</li>
    <li><strong>Rule-based ordering (Decision List)</strong>：将规则严格按优先级排列，测试实例顺序匹配，触发第一条符合的规则后即停止评估。WEKA 中的 JRip 生成的即为 Decision List。</li>
</ul>

<h2>3. WEKA 规则引擎：JRip 与 PART 核心算法流</h2>
<p>规则分类算法多采用 <strong>Separate-and-Conquer (分离并征服 / Sequential Covering)</strong> 策略。</p>
<p><strong>JRip (RIPPER) 的增量学习流:</strong></p>
<p><strong>输入：</strong>带有特征与类标的训练数据集。<br>
<strong>输出：</strong>有序的 IF-THEN 规则列表 (Decision List)。</p>
<ol>
    <li><strong>[初始化规则集]</strong>：创建空白的规则集。</li>
    <li><strong>[学习单条规则]</strong>：在当前剩余样本上，利用 <strong>FOIL 信息增益</strong> 逐步添加 <code>AND</code> 条件，生成覆盖某一类别最优的单条规则。</li>
    <li><strong>[MDL 停止准则]</strong>：在生成规则时，算法会通过 <strong>最小描述长度 (MDL)</strong> 监控模型复杂度。若新规则增加的准确率不足以弥补其造成的模型膨胀（通常设定为 64 bits 的惩罚线），则停止该规则的生长。</li>
    <li><strong>[移除已覆盖样本]</strong>：从训练集中将该规则覆盖的所有样本删除。</li>
    <li><strong>[循环迭代]</strong>：重复步骤2，针对剩余样本继续生成规则，直至满足终止条件。</li>
    <li><strong>[全局修剪]</strong>：进入 <strong>Global Optimization (全局修剪)</strong> 阶段，优化生成的决策列表。</li>
</ol>

<p><strong>PART 算法流程:</strong></p>
<p><strong>输入：</strong>带有特征与类标的训练数据集。<br>
<strong>输出：</strong>独立的 IF-THEN 规则集合。</p>
<ol>
    <li><strong>[构建局部决策树]</strong>：在当前全量数据上调用 C4.5 (J48) 构建 <strong>局部决策树 (Partial Decision Tree)</strong>。</li>
    <li><strong>[提取规则]</strong>：选取该树中覆盖样本最多、纯度最高的一条叶节点路径，将其转化为 <code>IF...THEN...</code> 规则。</li>
    <li><strong>[销毁决策树]</strong>：放弃并销毁这棵决策树 (Discard the tree)。</li>
    <li><strong>[移除并循环]</strong>：将提取到的规则覆盖的样本从数据集中删除。回到步骤1重复该过程，直到覆盖所有样本。该方法能提取高纯度规则，但计算开销较大。</li>
</ol>

<h3>🎯 经典例题</h3>
<p><strong>【Q1. 算法执行流程考点】</strong>请阐述 JRip（WEKA中的 RIPPER 算法）依赖什么底层停止机制来防止规则集过拟合？</p>
<p><strong>【解答】</strong>: JRip 依靠 <strong>MDL (Minimum Description Length, 最小描述长度)</strong> 原则作为停止准则。在逐步增加新规则或扩充规则前件时，算法会评估模型的描述长度。如果新规则带来的纯度提升不能抵消其造成的模型长度增加惩罚，系统会停止规则生长，以此控制模型复杂度并保证泛化能力。</p>
<p><strong>【Q2. 概念对比】</strong>PART 和 JRip 在生成“单条最优规则”时的策略有何本质区别？</p>
<p><strong>【解答】</strong>: JRip 采用增量生长策略，每次基于 FOIL 信息增益选择最优特征拼接 <code>AND</code> 条件；而 PART 的策略更为系统，它利用未被覆盖的样本构建一棵“局部决策树”，从树中提取出最优秀的一条路径作为规则，随后废弃该树。PART 通过局部建树确保了规则的高准确性，但计算成本明显高于 JRip。</p>
"""

m06 = """
<h1>MODULE 06: Ensemble Methods (集成学习)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>EnsembleMethods.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>Bias-Variance 偏差与方差的权衡、Bagging的Bootstrap机制、Random Forest特征随机化、Boosting(AdaBoost)的加权机制。</div>
<h2>1. 集成学习的核心思想 (Ensemble Philosophy)</h2>
<p>单一分类器常面临性能瓶颈：如浅层决策树通常具有较高的 <strong>偏差 (Bias)</strong>，容易欠拟合；而深层决策树则具有较高的 <strong>方差 (Variance)</strong>，容易对噪声产生过拟合。<br>
<strong>Ensemble (集成)</strong> 方法通过结合多个独立且具备 <strong>多样性 (Diverse)</strong> 的基分类器，能够有效平滑噪声以降低方差，或通过串联纠错机制降低偏差，从而显著提升泛化能力。</p>

<h2>2. 常见集成算法执行流</h2>
<p><strong>Bagging (Bootstrap Aggregation 装袋法)：降低方差 (Reduce Variance)</strong></p>
<p><strong>输入：</strong>原始数据集；基分类器数量 $M$。<br>
<strong>输出：</strong>集成后的强分类器预测结果。</p>
<ol>
    <li><strong>[并行启动]</strong>：对包含 $N$ 条样本的原始数据集，并行启动 $M$ 个训练过程。</li>
    <li><strong>[抽样组集]</strong>：每个过程通过 <strong>有放回抽样</strong> $N$ 次构建子数据集（约有 36.8% 的样本未被抽中，称为 OOB 数据）。</li>
    <li><strong>[独立训练]</strong>：各个过程独立训练一个不受限的高复杂度基分类器（如深层决策树）。</li>
    <li><strong>[多数投票]</strong>：测试时，所有基分类器对新样本进行预测，分类任务采用 <strong>多数投票 (Majority Voting)</strong>，回归任务取平均。平均效应能够抵消单个分类器的随机误差，大幅降低方差。</li>
</ol>

<p><strong>Random Forest (随机森林)：提升特征多样性</strong></p>
<p><strong>输入：</strong>原始数据集；决策树数量 $M$；特征子集大小 $K$。<br>
<strong>输出：</strong>随机森林集成模型。</p>
<ol>
    <li><strong>[继承抽样]</strong>：继承 Bagging 的 Bootstrap 抽样机制。</li>
    <li><strong>[屏蔽全量特征]</strong>：在构建每棵决策树的每一个分裂节点时，算法 <strong>不评估所有特征</strong>。</li>
    <li><strong>[随机特征子集]</strong>：而是随机抽取一个包含 $K$ 个特征的子集（通常 $K=\\log_2M+1$）。</li>
    <li><strong>[特征选择]</strong>：决策树仅能在该随机子集中选择最优特征进行分裂。这种特征层面的随机化进一步降低了树与树之间的相关性，提升了集成的 <strong>多样性 (Diversity)</strong>，使得平滑方差的效果更佳。</li>
</ol>

<p><strong>AdaBoost (Adaptive Boosting)：降低偏差 (Reduce Bias)</strong></p>
<p><strong>输入：</strong>原始数据集；弱分类器数量 $M$。<br>
<strong>输出：</strong>加权投票的强分类器模型。</p>
<ol>
    <li><strong>[初始化权重]</strong>：采用串行方式，训练一系列弱分类器（如单层决策树 Decision Stump）。初始时，所有样本被赋予相等的权重。</li>
    <li><strong>[基分类器训练]</strong>：第一棵基分类器进行训练后，必然会错误分类一部分难例样本。</li>
    <li><strong>[权重更新]</strong>：系统将这些被错误分类的样本的 <strong>权重提升 (Increase weight)</strong>，并降低分类正确样本的权重。</li>
    <li><strong>[串行修正]</strong>：后续的分类器被迫专注于拟合上一轮具有较高权重的错分样本。此过程迭代进行，逐步修正前期模型的偏差。</li>
    <li><strong>[加权投票]</strong>：测试时，所有分类器参与投票。系统根据各分类器在训练阶段的 <strong>错误率</strong> 为其分配权重 ($\\alpha_m = \\frac{1}{2} \\ln \\left(\\frac{1-\\epsilon}{\\epsilon}\\right)$)，错误率越低的分类器拥有越大的决策话语权。</li>
</ol>

<p><strong>Stacking (堆叠法):</strong></p>
<p><strong>输入：</strong>训练数据集；多种不同类型的初级基分类器；高阶元分类器。<br>
<strong>输出：</strong>Stacking 集成模型。</p>
<ol>
    <li><strong>[基础层预测]</strong>：在基础层使用多种不同类型的基分类器（例如决策树、KNN、朴素贝叶斯）。</li>
    <li><strong>[获取结果]</strong>：将测试样本输入这些基分类器，得到各自的预测结果。</li>
    <li><strong>[特征重组]</strong>：将这些预测结果组合成全新的特征向量。</li>
    <li><strong>[元分类器融合]</strong>：将新特征向量输入更高阶的 <strong>元分类器 (Meta Classifier)</strong>，由其学习如何结合这些底层预测，得出最终结论。</li>
</ol>

<h3>🎯 经典例题</h3>
<p><strong>【Q1. 加权公式推演题】</strong>在 AdaBoost 中，如果第 $m$ 个基分类器的加权错误率 $\\epsilon_m$ 等于 0.2，请写出分配给这棵树的权重 $\\alpha_m$ 的计算过程。</p>
<p><strong>【解答】</strong>:<br>
在 AdaBoost 中，单棵树的权重公式为：$\\alpha_m = \\frac{1}{2} \\ln \\left( \\frac{1 - \\epsilon_m}{\\epsilon_m} \\right)$。<br>
代入 $\\epsilon_m = 0.2$，得到 $\\alpha_m = \\frac{1}{2} \\ln \\left( \\frac{1 - 0.2}{0.2} \\right) = \\frac{1}{2} \\ln (4)$。<br>
因其错误率较低，该分类器将在最终投票中获得较高的决策权重。</p>
<p><strong>【Q2. 概念分析】</strong>为了解决过度深层决策树产生的过拟合问题，应倾向于选择 AdaBoost 还是 Random Forest 算法？为什么？</p>
<p><strong>【解答】</strong>: 应选择 <strong>Random Forest (随机森林) / Bagging 方法</strong>。<br>
极深的决策树属于**低偏差、高方差**（Low Bias, High Variance）模型。Bagging 通过 Bootstrap 采样与随机特征选取，集成多棵高方差树，利用平均效应有效降低整体方差 (Reduce variance)，解决过拟合问题。<br>相对而言，AdaBoost 的设计目标是**降低偏差**（Reduce bias），适用于提升简单弱分类器（高偏差）的表现。将过拟合的深树放入 AdaBoost 会进一步放大其对噪声的拟合，导致更严重的泛化失败。</p>
"""

m07 = """
<h1>MODULE 07: Evaluation & The Problem of Overfitting (过拟合与模型评估)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>Evaluation.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>泛化与过拟合的概念、经验风险最小化(ERM)理论、交叉验证及 63.2 Bootstrap 评估方法的数学原理与应用场景。</div>
<h2>1. 核心概念：Generalization (泛化) vs Overfitting (过拟合)</h2>
<p>模型训练的最终目的并非仅仅拟合历史数据，而是为了对 <strong>未来未见的数据 (Unseen / Future data)</strong> 具有良好的预测能力，这种能力被称为 <strong>Generalization (泛化能力)</strong>。<br>
<strong>Overfitting (过拟合)</strong> 指的是：当模型复杂度过高时，它可能记住了训练集中的特定噪声或离群点特征。虽然在训练集上表现优异，但在面对新样本时，预测性能显著下降。</p>
<h2>2. 经验风险最小化 (Empirical Risk Minimization, ERM) 理论</h2>
<p>理想状态下，模型应基于数据的“真实联合概率分布”以最小化“期望风险 (Expected Risk)”。<br>
然而，真实分布不可知，算法只能通过最小化在 <strong>有限抽样样本</strong> 上的错误率来进行优化，即 <strong>经验风险最小化 (ERM)</strong>。<br>
由于 ERM 仅针对已知的局部样本，如果算法缺乏正则化约束，就容易将样本中的偏差特征视为普遍规律，从而引发过拟合。</p>
<h2>3. 模型验证方法与数据分割</h2>
<p>为了客观评估模型泛化能力，需使用独立的测试数据。WEKA 提供了多种验证流程：</p>
<p><strong>Cross-Validation (交叉验证 / K-Fold CV):</strong></p>
<p><strong>输入：</strong>原始数据集；划分折数 $K$ (如 10)。<br>
<strong>输出：</strong>模型平均性能评估指标。</p>
<ol>
    <li><strong>[数据划分]</strong>：例如 10-Fold CV，将原始数据集划分为 <strong>10 个互不重叠的等离子集 (Folds)</strong>。</li>
    <li><strong>[分层机制]</strong>：通常采用 <strong>Stratification (分层机制)</strong>：确保每个划分块中的类别比例与原数据集一致，避免样本分布极度失衡。</li>
    <li><strong>[迭代训练]</strong>：进行 10 轮迭代训练。每轮选取第 $i$ 个块作为测试集 (Test Set)，其余 9 个块合并作为训练集。</li>
    <li><strong>[评估计算]</strong>：10轮结束后，计算这 10 次测试结果的平均值，以获得稳定可靠的性能评估。</li>
</ol>

<p><strong>63.2 Bootstrap 评估引擎:</strong></p>
<p><strong>输入：</strong>原始数据集 (规模为 $N$)。<br>
<strong>输出：</strong>基于 OOB 测试集的模型性能评估指标。</p>
<ol>
    <li><strong>[适用场景]</strong>：在数据集较小不适合进行交叉验证时，可使用 Bootstrap 方法。</li>
    <li><strong>[有放回抽样]</strong>：采用 <strong>有放回随机抽样 (Sampling with replacement)</strong>，抽取与原数据集规模 $N$ 相同次数的样本构成训练集。</li>
    <li><strong>[概率推导]</strong>：在一次有放回抽样中，特定样本未被抽中的概率为 $\\left(1 - \\frac{1}{N}\\right)$。在 $N$ 次抽样中始终未被抽中的概率为 $\\left(1 - \\frac{1}{N}\\right)^N$。当 $N \\to \\infty$ 时，该概率趋近于 $\\frac{1}{e} \\approx 0.368$。</li>
    <li><strong>[训练集构成]</strong>：这意味着训练集中由于重复抽样，实际只包含约 <strong>63.2% 的独特原始样本</strong>。</li>
    <li><strong>[OOB 测试集]</strong>：剩下的约 <strong>36.8% 未被抽中的样本构成 Out-of-Bag (OOB) 集合</strong>，将其作为无偏的测试集用于评估模型性能。</li>
</ol>
<h3>🎯 经典例题</h3>
<p><strong>【Q1. 计算推演题】</strong>某研究机构收集了 100 份罕见病样本。为避免传统 Holdout 方法导致测试集过小，决定采用 63.2% Bootstrap 策略。请问需执行多少次“有放回抽样”以组建训练集？生成的训练集中预期包含多少条不重复的独特样本？未被抽中的样本有何用途？</p>
<p><strong>【解答】</strong>:<br>
- 需执行与原始样本量相等的有放回抽样次数，即 <strong>100次</strong>。<br>
- 根据数学期望，经过抽样后，训练集预期包含约 $63.2\\%$ 的独特样本，即 <strong>约 63 条独立样本</strong>（其余为重复样本）。<br>
- 约 36.8% 始终未被抽中的数据构成 <strong>Out-of-Bag (OOB) 样本</strong>，作为无偏的 <strong>测试集</strong>，用于最终评估模型性能。</p>
"""

m08 = """
<h1>MODULE 08: Frequent Pattern Analysis (Apriori Algorithm)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>FrequentPatternAnalysis_AP.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>支持度与置信度的逻辑、Apriori 反单调性及候选生成与剪枝机制、极大项集与闭合项集概念、Lift 提升度的作用。</div>
<h2>1. 市场篮子分析 (Market Basket Analysis)</h2>
<p>这是一种经典的无监督模式挖掘方法，旨在从交易记录 (Transaction Databases) 中找出 <strong>频繁且同时出现</strong> 的物品组合，常用于零售优化与推荐系统。</p>
<ul>
    <li><strong>Support (支持度)</strong>：项集 $X$ 在所有交易记录中出现的概率，反映组合的频繁程度：$Support(X) = \\frac{Count(X)}{Total\\ Transactions}$。</li>
    <li><strong>Confidence (置信度)</strong>：条件概率 $P(Y|X)$，即包含 $X$ 的交易中同时包含 $Y$ 的比例：$Confidence(X \\Rightarrow Y) = \\frac{Support(X \\cup Y)}{Support(X)}$。用于生成关联规则。</li>
</ul>
<h2>2. Apriori 核心原理：反单调性 (Anti-monotone)</h2>
<p>为了应对组合爆炸问题，算法引入了 <strong>Apriori 先验性质 (反单调性)</strong>：<br>
<strong>定理：一个频繁项集的所有非空子集也必须是频繁的。</strong><br>
<strong>剪枝推论</strong>：如果某一项集不满足最小支持度（即非频繁），则任何包含该项集的超集也必定非频繁，因此可将其从候选搜索空间中直接剔除。</p>

<h2>3. Apriori 算法流程</h2>
<p>Apriori 算法通过逐层迭代发掘频繁项集：</p>
<p><strong>输入：</strong>交易记录数据库；最小支持度阈值 <code>Min_Support</code>。<br>
<strong>输出：</strong>所有满足支持度阈值的频繁项集。</p>
<ol>
    <li><strong>[扫描数据库]</strong>：统计所有单一商品的频率，剔除不满足 <code>Min_Support</code> 的项，生成 <strong>频繁 1-项集 ($L_1$)</strong>。</li>
    <li><strong>[候选生成]</strong>：通过上一代的频繁集 $L_{k-1}$ 相互拼接，生成高一维度的 <strong>候选集 $C_k$</strong>。</li>
    <li><strong>[剪枝策略]</strong>：检查 $C_k$ 中每个候选集的所有 $(k-1)$ 级子集，如果发现有子集不在 $L_{k-1}$ 中，根据反单调性，立刻将该候选集从 $C_k$ 中删除。</li>
    <li><strong>[支持度验证]</strong>：对剪枝后剩余的 $C_k$ 候选项，再次扫描数据库计算实际支持度，满足阈值的项集成为 <strong>$L_k$</strong>。</li>
    <li><strong>[循环迭代]</strong>：重复步骤 2 至 4，直到无法生成新的频繁项集。</li>
</ol>

<h2>4. 项集的存储与压缩技术 (Maximal & Closed)</h2>
<p>为了解决高维频繁模式导致子集呈指数级增长的存储问题，通常采用以下压缩表示：</p>
<ul>
    <li><strong>极大频繁项集 (Maximal frequent itemsets)</strong>：自身是频繁的，且没有任何真超集 (proper superset) 是频繁的。它隐含了其所有子集均频繁的信息。</li>
    <li><strong>闭合项集 (Closed Patterns)</strong>：自身是频繁的，且不存在与它拥有 <strong>相同支持度计数 (count)</strong> 的真超集。它在压缩数据的同时，保留了所有子集的确切支持度信息。</li>
</ul>
<h2>5. 规则评估：Confidence 的局限与 Lift</h2>
<p>高 Confidence 可能产生统计学上的误导。<br>
例如：规则 <code>玩游戏 -> 买手机</code> 置信度达 90%。但如果人群中购买手机的基础概率本身就是 95%，这说明“玩游戏”实际上降低了买手机的概率，高置信度为虚假的正相关。<br>
为识别真实的关联，需引入 <strong>Lift (提升度)</strong>：$Lift(X, Y) = \\frac{P(X \\cup Y)}{P(X) \\times P(Y)}$。只有当 $Lift > 1$ 时，才表明 $X$ 对 $Y$ 具有实质的正向促进作用。</p>

<h3>🎯 经典例题</h3>
<p><strong>【Q1. 指标计算题】</strong>在拥有 100 条记录的数据库中，单买面包的记录有 40 条，单买黄油的有 50 条，两者同时购买的有 30 条。<br>
请计算规则 <code>{面包} -> {黄油}</code> 的 Support, Confidence，并计算 Lift 以说明这是否是一条有效的正向关联规则。</p>
<p><strong>【解答】</strong>:<br>
- <strong>Support</strong> = $\\frac{30}{100} = 30\\%$<br>
- <strong>Confidence</strong> = $\\frac{Support(面包 \\cup 黄油)}{Support(面包)} = \\frac{30}{40} = 75\\%$。<br>
- <strong>Lift (提升度)</strong> = $\\frac{P(面包 \\cup 黄油)}{P(面包) \\times P(黄油)} = \\frac{0.3}{0.4 \\times 0.5} = 1.5$。<br>
- 结论：因为 <strong>Lift = 1.5 > 1</strong>，表明购买面包对购买黄油有显著的正向促进作用，为有效关联规则。</p>
<p><strong>【Q2. 剪枝策略填空题】</strong>在 Apriori 算法生成候选集 $C_3 = \\{苹果, 橙子, 香蕉\\}$ 时，系统会检查其所有的 2 维子集：$\\_{______}\\_$，$\\_{______}\\_$，$\\_{______}\\_$。若发现有任何子集未出现在 $L_2$ 中，则会将其剪枝。这利用了算法的 $\\_{______}\\_$ 性质。</p>
<p><strong>【解答】</strong>: 填入 <code>{苹果, 橙子}</code>、<code>{苹果, 香蕉}</code>、<code>{橙子, 香蕉}</code>。该机制基于 <code>Anti-monotone (反单调性)</code> 性质：频繁项集的所有子集必须频繁。</p>
"""

m09 = """
<h1>MODULE 09: Partitioning & Hierarchical Methods (划分与层次聚类)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>ClusterAnalysis_Part.pptx</code> & <code>ClusterAnalysis_Hier.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>划分聚类的原理、K-Means 迭代机制及其在非凸形状和离群点前的局限性、层次聚类的三种 Linkage 距离计算及其不可逆特性。</div>
<h2>1. 聚类分析概述 (Clustering)</h2>
<p>聚类属于无监督学习，目的在于根据数据实例的内部特征（如距离度量），将数据集划分为多个簇，使得 <strong>簇内高度相似（内聚），簇间显著不同（分离）</strong>。</p>

<h2>2. 划分聚类方法：K-Means (SimpleKMeans)</h2>
<p>K-Means 是最典型的中心点划分算法，其迭代步骤如下：</p>
<p><strong>输入：</strong>数据集；目标簇数量 $K$。<br>
<strong>输出：</strong>划分为 $K$ 个簇的数据点集合及各簇质心。</p>
<ol>
    <li><strong>[初始化质心]</strong>：随机选择 $K$ 个数据点作为初始簇质心 (Centroids)。</li>
    <li><strong>[样本分配]</strong>：计算每个数据点到各质心的欧氏距离，将数据点分配至距离最近的质心所在的簇 $C_i$。</li>
    <li><strong>[质心更新]</strong>：计算每个簇内所有样本坐标的 <strong>几何平均值 (均值, $\\mu_i$)</strong>，将质心更新为该均值点位置。</li>
    <li><strong>[收敛停止]</strong>：上述分配与更新步骤循环进行，直至质心位置不再变化，或整体误差平方和 <strong>(SSE) 达到收敛</strong>，算法停止。</li>
</ol>

<h3>K-Means 的主要局限性：</h3>
<ul>
    <li><strong>对初始中心敏感</strong>：随机初始化可能导致算法陷入 <strong>局部最优解 (Local Optimum)</strong>。通常需要结合 K-Means++ 或多次随机重启来优化初始点选择。</li>
    <li><strong>难以发现非凸形簇 (Non-convex bias)</strong>：K-Means 的距离和均值机制假定簇呈现球形分布。面对环状、S形等复杂拓扑结构的簇，它无法进行合理划分。</li>
    <li><strong>对同心或尺寸差异过大的簇处理不佳</strong>：如果两个簇中心接近（如内部小簇嵌套在外部大环内），算法会执行不合理的强制切割。</li>
    <li><strong>对离群噪点 (Outliers) 敏感</strong>：极端的噪点在计算均值时会显著偏移质心位置。<br>
    <strong>改进方法：K-Medoids 算法 (PAM)</strong>，使用簇内实际存在的中位数样本点作为代表，降低了离群点对重心的影响。</li>
</ul>

<h2>3. 层次聚类方法 (Hierarchical Clustering)</h2>
<p>层次聚类无需预先指定 $K$ 值，通常采用自底向上的 <strong>凝聚 (Agglomerative, AGNES)</strong> 策略。初始时每个数据点自成一簇，在每轮迭代中合并距离最近的两个簇，最终构建出一棵 <strong>层次聚类树 (Dendrogram)</strong>。</p>
<h3>簇间距离的计算方法 (Linkage)：</h3>
<ul>
    <li><strong>Single-linkage (单链法 / 最短距离)</strong>：定义两个簇的距离为其成员间 <strong>最小</strong> 的距离：$d(A, B) = \\min_{x \\in A, y \\in B} d(x, y)$。<br>
    <strong>缺陷</strong>：易引发 <strong>链式效应 (Chaining phenomenon)</strong>，即由于少量噪点作为桥梁，将本不相干的群体合并，偏向于生成细长形状的簇。此过程与最小生成树 (MST) 等价。</li>
    <li><strong>Complete-linkage (全链法 / 最长距离)</strong>：定义两个簇的距离为其成员间 <strong>最大</strong> 的距离：$d(A, B) = \\max_{x \\in A, y \\in B} d(x, y)$。倾向于生成紧凑的球形簇，但易受边缘离群点影响。</li>
    <li><strong>Ward's Method (沃德方法)</strong>：基于方差分析，优先合并使得整体误差平方和（系统变异度）增加最小的两个簇。</li>
</ul>
<p><strong>不可逆性 (Irreversible)</strong>：层次聚类的一大缺陷在于，合并决策一旦做出便无法撤销。早期的局部错误合并会一直传导至最终结果，无法通过后续步骤修正。</p>

<h3>🎯 经典例题</h3>
<p><strong>【Q1. 算法执行演练题】</strong>简述 K-Means 算法在每轮迭代中执行的两个数学步骤，并说明其收敛停止条件。</p>
<p><strong>【解答】</strong>:<br>
- <strong>分配步骤 (Assignment)</strong>：计算每个数据点 $x$ 到当前所有质心 $\\mu$ 的距离，将其分配至最近的质心所在的簇：$C_i^{(t)} = \\{x : ||x - \\mu_i^{(t)}|| \\le ||x - \\mu_j^{(t)}||\\}$。<br>
- <strong>更新步骤 (Update)</strong>：对每个簇 $C_i$，重新计算其中所有点的均值坐标，作为新的质心：$\\mu_i^{(t+1)} = \\frac{1}{|C_i^{(t)}|} \\sum_{x \\in C_i^{(t)}} x$。<br>
- <strong>收敛条件</strong>：当最新一次迭代后，所有质心坐标不再改变（即簇分配结果稳定，SSE不再下降），算法宣告收敛并停止。</p>
<p><strong>【Q2. 概念辨析】</strong>对于呈现同心圆状分布的空间数据集（内部球形核心，外部环形分布），若使用 Single-linkage 与 K-Means 分别聚类，结果有何不同？</p>
<p><strong>【解答】</strong>:<br>
- <strong>Single-linkage</strong>：只要外环内部样本分布紧密，且内外层之间存在较宽的间隙，单链法能够通过最近点连接原则正确地将内外两层剥离成不同簇。<br>
- <strong>K-Means</strong>：由于两类分布共享相同的中心区域，K-Means 会因追求类内方差最小化，而用线性边界将同心圆强行切割为多块（例如左右两半），导致聚类失效。</p>
"""

m10 = """
<h1>MODULE 10: Density-Based Clustering (基于密度的聚类)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>ClusterAnalysis_DB.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>DBSCAN 算法中三类实体（核心点、边界点、噪声点）的定义、密度相连的概念、应对任意形状与抗噪性的原理、以及 OPTICS 应对变密度分布的改进。</div>
<h2>1. 基于密度聚类的理念</h2>
<p>K-Means 假设簇为凸形分布，层次单链法易受噪声影响。而 <strong>基于密度的聚类 (Density-Based Clustering)</strong> 认为：聚类结构是被低密度区域分隔开的“高密度连通区域”。只要样本分布的密度符合阈值并保持连通，即可构成一个簇，从而能够识别任意非凸形状并有效过滤噪声。</p>

<h2>2. DBSCAN 算法原理与执行流程</h2>
<p>DBSCAN 依赖两个核心参数：<br>
1. <strong>Eps ($\\epsilon$)</strong>：邻域半径，即距离探测阈值。<br>
2. <strong>MinPts</strong>：核心点在其 Eps 邻域内需包含的最少样本数要求。</p>
<h3>数据点的三种类型划分：</h3>
<ul>
    <li><strong>Core points (核心点)</strong>：若某样本的 Eps 邻域内样本数 $\\ge MinPts$，则为核心点。</li>
    <li><strong>Border points (边界点)</strong>：邻域内样本数 $< MinPts$，不满足核心点要求，但其位于某个核心点的 Eps 邻域内部，被划分为簇的边缘部分。</li>
    <li><strong>Noise (噪声点)</strong>：既非核心点也不是边界点，无法归入任何簇的样本。这一机制赋予了 DBSCAN 优良的抗噪性。</li>
</ul>
<h3>DBSCAN 的簇扩展流程：</h3>
<p><strong>输入：</strong>数据集；邻域半径 $Eps$；最小样本数 $MinPts$。<br>
<strong>输出：</strong>基于密度的聚类簇及噪声点集合。</p>
<ol>
    <li><strong>[初始化状态]</strong>：初始化所有数据点为未访问状态 (Unvisited)。</li>
    <li><strong>[选取起始点]</strong>：随机选取一个未访问点 $A$，标记为已访问。并获取其 Eps 邻域。</li>
    <li><strong>[判定噪声]</strong>：若邻域样本数 $< MinPts$，将 $A$ 标记为 <strong>Noise</strong>（后续可能被重置为边界点）。</li>
    <li><strong>[构建新簇]</strong>：若邻域样本数 $\\ge MinPts$，$A$ 成为核心点，创建一个新簇 $C$。将邻域内的点加入 $C$ 的考察队列。</li>
    <li><strong>[密度可达扩展]</strong>：遍历考察队列中的点，若其邻域同样满足核心点条件，则将其邻域内的点也吸纳进队列，通过 <strong>密度相连 (Density-connectedness)</strong> 效应不断扩张簇的规模。</li>
    <li><strong>[循环迭代]</strong>：簇扩展停止后，重复选择未访问的点，直至所有点均被评估完毕。</li>
</ol>

<h2>3. 变密度数据的局限与 OPTICS 算法</h2>
<p><strong>DBSCAN 的局限性</strong>：它对全局固定的 Eps 和 MinPts 参数高度敏感。对于具有多种不同密度分布的数据集，单一的阈值难以同时识别高密度簇与低密度簇。</p>
<h3>OPTICS (Ordering Points To Identify the Clustering Structure)</h3>
<p>OPTICS 通过构建数据点的一维排序输出，避免了显式生成聚类，并引入了动态距离的概念：<br>
1. <strong>Core-distance (核心距离)</strong>：使某点恰好满足核心点条件（即包含 MinPts 个点）所需的最小邻域半径。公式：$Core\\_distance(p)$。<br>
2. <strong>Reachability-distance (可达距离)</strong>：某点 $o$ 从核心点 $p$ 处“被可达”的距离。它不能小于 $o$ 自身的核心距离：$Reachability\\_distance(p, o) = \\max(Core\\_distance(o), d(p, o))$。</p>
<p>基于此，OPTICS 可绘制 <strong>Reachability Plot (可达性图)</strong>。<br>
图表中的波谷 (Valleys) 表示距离紧凑的密集簇，而高峰表示稀疏区域或噪声点。通过观察可达性图的起伏，分析人员可以识别不同密度级别的聚类结构，有效解决了全局阈值难以适配变密度数据的难题。</p>

<h3>🎯 经典例题</h3>
<p><strong>【Q1. 实体判定题】</strong>执行 DBSCAN，参数 $Eps=1.5$, $MinPts=4$。侦测到点 A 的 1.5 邻域内只有 2 个点（点 A 和点 B）。而点 B 邻域内有 10 个点。请判断点 A 和点 B 的实体类别。</p>
<p><strong>【解答】</strong>: <br>
- 点 B 邻域内包含 10 个点 $\\ge MinPts$，因此点 B 是 <strong>Core point (核心点)</strong>。<br>
- 点 A 邻域内仅 2 个点 $< MinPts$，不具备成为核心点的资格。但它被包含在核心点 B 的 1.5 邻域内，因此被判定为 <strong>Border point (边界点)</strong>。</p>
<p><strong>【Q2. 算法原理题】</strong>如果将 DBSCAN 的 $MinPts$ 设置为 1，此时算法实质上退化为了何种经典聚类方法？请说明理由。</p>
<p><strong>【解答】</strong>: 会退化为 <strong>Single-linkage Agglomerative (单链法层次聚类)</strong> 结合距离截断。<br>
当 $MinPts=1$ 时，任何样本点只需包含自己便能满足核心点条件，此时所有点均成为核心点，噪声排除机制失效。随后，只要两点之间的距离 $\\le Eps$，它们便会合并为同一簇。最终产生的连通图等同于在构建最小生成树时于 $Eps$ 处截断的结果，与单链法的链式效应如出一辙。</p>
"""

m11 = """
<h1>MODULE 11: Cluster Analysis Evaluation (聚类评估)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>ClusterAnalysis_EV.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>聚类分析的有效性前提、Elbow法则探查簇数量的逻辑、Silhouette轮廓系数评估聚类质量的方法、以及基于外部标签的评估指标。</div>
<h2>1. 聚类评估的三个层次</h2>
<p>评估无监督聚类任务面临缺乏 Ground Truth 的挑战，通常包含以下三个层面：</p>
<ul>
    <li><strong>Tendency (聚类趋势)</strong>：验证数据集本身是否存在非随机的聚类结构（可通过霍普金斯统计量等方法检验）。若数据仅为均匀分布的随机噪声，则聚类毫无意义。</li>
    <li><strong>Quantity (簇数量探查)</strong>：确定数据集包含的最优簇数量 $K$。</li>
    <li><strong>Quality (聚类质量)</strong>：衡量聚类结果是否达到了“簇内相似度高、簇间差异性大”的标准。</li>
</ul>
<h2>2. 内在评估指标 (Intrinsic Measures)</h2>
<p>在缺乏真实标签的情况下，依赖数据内在几何分布进行验证：</p>
<ul>
    <li><strong>Elbow Method (肘部法则) 探寻 K 值</strong>：<br>
    通过绘制误差平方和 (SSE) 随 K 值变化的曲线。当 K 值增大时，SSE 必然递减。<br>
    如果在某个特定的 K 值处，SSE 的下降幅度由急剧转为平缓，曲线呈现出弯曲的 <strong>“肘部 (Elbow)”</strong>，则认为该点为较优的分类数量。<br>
    <strong>局限性</strong>：在实际数据中，类间界限往往模糊，曲线平滑无明显拐点，导致此法则难以应用。</li>
    
    <li><strong>Silhouette Coefficient (轮廓系数) 评估聚类质量</strong>：<br>
    考察单一样本点 $i$ 的聚类合理性：<br>
    1. 计算 $i$ 到其 <strong>所属簇</strong> 内所有其他点的平均距离 $a(i)$，表征内聚度。<br>
    2. 计算 $i$ 到其他 <strong>相邻簇</strong> 内所有点的平均距离，取其中的最小值 $b(i)$，表征分离度。<br>
    计算公式：$s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}$。<br>
    <strong>指标解释</strong>：数值范围为 <strong>[-1, 1]</strong>。<br>
    - 若结果 <strong>接近 1</strong>，表明 $b \\gg a$，样本与同簇极近且远分于他簇，聚类质量优异。<br>
    - 若结果 <strong>接近 0</strong>，表明 $a \\approx b$，样本位于两簇边界，归属不明确。<br>
    - 若结果 <strong>接近 -1</strong>，表明 $a \\gg b$，样本距离自身簇反而比距离别簇更远，说明聚类分配错误。</li>
</ul>
<h2>3. 外在评估指标 (Extrinsic Measures)</h2>
<p>当存在已知真实分类标签时，可将其作为基准进行外部评估：</p>
<ul>
    <li><strong>B-Cubed 评估法</strong>：<br>
    针对样本级别的配对精确度：针对某个样本，其所属簇中真实类别与其相同的样本数比例为其 Precision；整体数据集中真实类别相同且被聚入同一簇的样本数比例为其 Recall。计算全员的加权平均得分。</li>
    <li><strong>Classes to clusters evaluation (类别到簇映射)</strong>：<br>
    面对聚类算法输出的匿名簇编号（如 Cluster 0, Cluster 1），WEKA 使用贪心算法寻找最优的映射关系，使得预测簇编号与真实类别标签的重合样本数量最大化。<br>
    完成匹配后，未对齐的样本被视为误分类实例，据此计算等效的 <strong>分类错误率 (Error Rate)</strong>。</li>
</ul>
<h3>🎯 经典例题</h3>
<p><strong>【Q1. 轮廓系数计算题】</strong>在聚类任务中，已知数据点 $x$ 到同簇其它样本的平均距离为 5，到距离其最近的相邻簇中样本的平均距离为 1。请计算此样本的轮廓系数，并判定该点是否被正确聚类。</p>
<p><strong>【解答】</strong>:<br>
- 内聚距离 $a=5$，分离距离 $b=1$。<br>
- 轮廓系数公式：$s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))} = \\frac{1 - 5}{\\max(5, 1)} = \\frac{-4}{5} = -0.8$<br>
- <strong>判定</strong>：轮廓系数为 $-0.8$，接近 $-1$。说明该样本与所属簇内的节点差异巨大，反而更贴近其他簇，表明算法在此处做出了严重的错误划分。</p>
<p><strong>【Q2. 评估机制解析】</strong>使用具备 <code>Real_Labels</code> 的数据集在 WEKA 中运行 K-Means。输出结果显示 <code>Incorrectly clustered instances : 15.0%</code>。WEKA 是如何将无意义的簇编号转化为准确错误率的？</p>
<p><strong>【解答】</strong>: WEKA 采用了 <strong>Classes to clusters evaluation (类别到簇匹配映射)</strong> 机制。<br>
在算法输出任意名称的簇别后，WEKA 会尝试所有排列组合，找到使预测簇与真实大类之间重叠对齐样本数最大的最佳映射对应。<br>
在确定映射关系后，未能正确落入对应真实类别映射圈内的游离样本，即被等效统计为分类错误样本，由此推算出直观的 15.0% 错误率。</p>
"""

m12 = """
<h1>MODULE 12: Data Warehouse & OLAP (数据仓库与多维分析)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>DataWarehouse_DCOLAP.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>OLTP与OLAP架构差异、基于维度与事实的星型建模、Lattice晶格组合、以及四种 OLAP 多维查询操作 (Roll-up, Drill-down, Slice, Dice)。</div>
<h2>1. OLTP 与 OLAP 系统的差异</h2>
<p><strong>OLTP (联机事务处理)</strong> 面向日常业务操作，支持高并发的插入、更新和删除事务，优化数据一致性。若在其上执行涉及全库扫描的大规模聚合查询，会消耗大量资源并可能锁定表。<br>
为满足企业级报表与分析需求，<strong>Data Warehouse (数据仓库)</strong> 应运而生。数据仓库基于 <strong>OLAP (联机分析处理)</strong> 架构，通过预先计算汇总历史数据，存储在多维的 Data Cube (数据立方体) 中，实现对复杂聚合查询的毫秒级响应。</p>
<h2>2. 维度建模 (Dimension Modeling) 与 Lattice 结构</h2>
<p>数据仓库常采用星型模型 (Star Schema)，其包含两类基本表：</p>
<ul>
    <li><strong>Dimension (维度)</strong>：分析观察业务的角度与层级，如时间、地点、产品等，提供数据过滤和分组依据。</li>
    <li><strong>Fact / Measure (事实与度量)</strong>：被量化分析的核心业务指标数值，如销售量、金额等。</li>
</ul>
<p><strong>Lattice Structure (晶格网络)</strong>：<br>
包含 $N$ 个维度的 Data Cube 会生成 $2^N$ 种不同聚合程度的 <strong>Cuboids (立方体视图)</strong>。它们从包含全维度的最精细基础视图 (Base Cuboid) 逐层向上归总，直到 0-D 没有任何维度限制的 <strong>Apex Cuboid (全量汇总)</strong>，整体构成了具有偏序关系的晶格层级网络。</p>
<h2>3. 四大核心 OLAP 操作</h2>
<p>基于预定义的概念层级 (Concept Hierarchies，如：日 $\\rightarrow$ 月 $\\rightarrow$ 年)，用户可在多维数据模型中进行灵活的交互式分析：</p>
<ul>
    <li><strong>Roll-up (上卷 / 归约)</strong>：<br>
        动作：消除某个细分维度，或沿维度概念层级向上合并数据。<br>
        效果：使数据从微观趋向宏观，汇总粒度变粗（如将每日销量上卷为按月统计）。</li>
    <li><strong>Drill-down (下钻 / 细化)</strong>：<br>
        动作：引入新的切分维度，或沿维度层级向下分解数据。<br>
        效果：提供更详细的微观视图，数据粒度变细（如将某省份数据拆解至各个城市）。</li>
    <li><strong>Slice (切片)</strong>：<br>
        动作：在单一维度上固定一个特定数值条件。<br>
        效果：从多维立方体中截取一张二维平面的数据切片（例如限制查询条件为 <code>"年份 = 2025年"</code>）。</li>
    <li><strong>Dice (切块)</strong>：<br>
        动作：同时在多个维度上划定取值范围进行条件约束。<br>
        效果：从完整立方体中抽取特定的子数据块（例如查询 <code>"年份 = 2025"</code> 且 <code>"区域 = 华南"</code> 且 <code>"类别 = 电子产品"</code> 的数据）。</li>
</ul>
<h3>🎯 经典例题</h3>
<p><strong>【Q1. 操作判定题】</strong>在记录 (Time, Location, Item) 三维数据的系统中，判断以下查询需求对应哪一种 OLAP 操作：<br>
A：不看月份明细与产品型号，直接汇总全年的总营业额。<br>
B：提取广州和深圳第一季度手机品类的具体销售数据报表。</p>
<p><strong>【解答】</strong>:<br>
- <strong>A 对应 Roll-up (上卷)</strong>。查询摒弃了地点与产品细分维度，并将时间沿概念层级上卷汇总至全年，展现了宏观数据。<br>
- <strong>B 对应 Dice (切块)</strong>。该查询在三个维度上均进行了条件限定（地点限定广深，时间限定一季度，商品限定手机），属于在多维空间中提取局部子块的操作。</p>
"""

m13 = """
<h1>MODULE 13: Data Cube Computation (数据立方体计算)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>DataCubeComputation_BUC.pptx</code> & <code>DataCubeComputation_CCIC.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>聚合函数性质分类、维数组合导致状态爆炸及 Iceberg Cube 机制，BUC 算法如何利用先验单调性执行自顶向下计算并进行底层剪枝。</div>
<h2>1. 聚集函数 (Aggregate Functions) 的计算性质</h2>
<p>在底层数据聚合推导高层指标时，聚合函数展现出不同的数学性质：</p>
<ul>
    <li><strong>Distributive (分配型)</strong>：<br>
    包含 <code>Count</code>, <code>Sum</code>, <code>Min</code>, <code>Max</code>。该类型函数能够直接利用子节点的聚合结果再次聚合，推导出父节点值（例如将几个城市的总销售额相加得出全省总额），计算效率极高。</li>
    <li><strong>Algebraic (代数型)</strong>：<br>
    如 <code>Average</code>。不能直接对子节点均值进行再平均，必须借助多个分配型指标（如保存各自的 <code>Sum</code> 和 <code>Count</code>）通过代数运算 $\\frac{\\sum Sum_i}{\\sum Count_i}$ 进行还原计算。</li>
    <li><strong>Holistic (整体型)</strong>：<br>
    如 <code>Median (中位数)</code>、<code>Mode (众数)</code>、<code>Rank</code>。子节点的聚合值无法用于推导父节点值，求全局中位数必须重新加载全局的底层原始记录重新排序，对算力消耗极大。</li>
</ul>
<h2>2. 维度诅咒与 Iceberg Cube (冰山立方体)</h2>
<p>对于高维稀疏数据集，构建完整的 Data Cube 会导致子节点数量呈 $2^N$ 指数爆炸。然而，许多细碎条件组合产生的数据记录数 $Count \\le 1$，不具备统计分析意义。<br>
<strong>Iceberg Cube (冰山立方体)</strong> 机制通过设置 <strong>Iceberg condition (阈值限制，通常为 Min_Support 频次门槛)</strong>，强制规定只有满足频率要求的高价值组合节点才会被计算与存储。低于阈值的长尾组合被视作冰山下的部分予以舍弃，大幅度削减计算量。</p>

<h2>3. BUC 算法机制与剪枝奥义 (Bottom-Up Construction)</h2>
<p>尽管名为 Bottom-Up，BUC 算法在构建 Data Cube 时实际采用 <strong>自顶向下 (从覆盖全集的 0-D Apex 节点开始，逐步向下深钻进行维度细化切割)</strong> 的执行流向。</p>
<p><strong>BUC 单调性剪枝 (Pruning) 的原理</strong>：<br>
在进行向下细分推导时，任何父集节点的总数量指标 (如 <code>Count</code>) 必然大于等于其派生的所有子集节点的数量。这称为 <strong>聚合频次的单调递减性</strong>。<br>
基于此特性，在自顶向下的递归深钻过程中，一旦某过渡节点统计出的 <code>Count</code> 值已经低于设定的 <strong>冰山生存阈值 (Iceberg Threshold)</strong>，则由它继续细分产生的所有子代节点，其数据量必定不可能回升超过该阈值。<br>
算法在此处直接触发 <strong>剪枝 (Prune)</strong>，立刻停止对该分支后续所有下层维度的组合探索与计算。这种及时的截断处理正是 BUC 在高维稀疏数据上性能卓越的根基。</p>

<h2>4. CCIC (Closed Cube) 的存储优化压缩</h2>
<p>对于紧密绑定的属性关联组合，向下细分的新维度可能并不会实质性削减数据量（例如特定设备的特定玩家群体，其汇总 <code>Count</code> 与上层指标完全相同）。<br>
<strong>Closed Cube</strong> 技术识别这种具备 <strong>Ancestor-descendant (祖先后代属性冗余)</strong> 关联的等值聚合，只存储最具代表性的宏观边界指标 <strong>闭合单元 (Closed cell)</strong>，跳过重复存储值，进一步达到极致的内存压缩效果。这常使用 <strong>Hass diagram (哈斯图)</strong> 展示其等价类包含关系。</p>
<h3>🎯 经典例题</h3>
<p><strong>【Q1. 函数性质判定题】</strong>如果在数据仓库设计中，需要实现跨层级宏观汇总。现面临三个指标：总样本数 (Count)，平均交易价 (Average) 和最长耗时 (Max)。请判定它们的聚合性质，并说明是否可借由子节点的局部指标合并得出全局结果。</p>
<p><strong>【解答】</strong>:<br>
- <strong>Max (最大值) -> Distributive (分配型)</strong>：可直接提取子节点最大值再次对比得出全局最大值。<br>
- <strong>Count (总点数) -> Distributive (分配型)</strong>：子节点数值直接累加即得整体数值。<br>
- <strong>Average (均值) -> Algebraic (代数型)</strong>：不能简单累加，需依赖子节点保存的 <code>Sum</code> 与 <code>Count</code> 值进行重算方可得出总体均值。</p>
<p><strong>【Q2. BUC 命名辨析】</strong>BUC 算法自称为 Bottom-Up (自底向上)，但实际构建过程是从涵盖全集的 Apex 节点向底层拆分解构的，为何采用该命名？</p>
<p><strong>【解答】</strong>: 算法名称源于其构建 Cuboid 维格体系的视角。它从维度数量最少（0维的 Apex 星核）开始计算，逐步增加维度条件向外扩展，不断构建出涉及维度更高、视图网格体积更庞大的底层 Base Cuboid。由于其构筑维度框架是从微小体量逐层向上叠加生长的，故命名为自底向上构建。</p>
"""

m14 = """
<h1>MODULE 14: 必考计算题公式汇编</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>综合提取自全部幻灯片</code></p>
<div class="highlight"><strong>专题说明：</strong>汇编各章节涉及公式及推演逻辑，提供计算复习指引。</div>
<h2>1. 混淆矩阵与性能评估指标</h2>
<ul>
    <li><strong>定位：</strong> 行代表 Actual (真实)，列代表 Predicted (预测)。</li>
    <li><strong>Accuracy (准确率):</strong> $\\frac{TP+TN}{TP+TN+FP+FN}$ （在类别失衡时参考性差）</li>
    <li><strong>Precision (精确率):</strong> $\\frac{TP}{TP+FP}$ （预测为正例中真实的比例）</li>
    <li><strong>Recall / TPR (召回率):</strong> $\\frac{TP}{TP+FN}$ （真实正例中被正确预测的比例）</li>
    <li><strong>Specificity / TNR:</strong> $\\frac{TN}{TN+FP}$ （真实负例被正确预测的比例）</li>
    <li><strong>NPV:</strong> $\\frac{TN}{TN+FN}$ （预测为负例中真负例的比例）</li>
    <li><strong>F-measure (F1-Score):</strong> $2 \\times \\frac{Precision \\times Recall}{Precision + Recall}$</li>
</ul>
<h2>2. 决策树分裂与纯度指标</h2>
<ul>
    <li><strong>Shannon's Entropy (信息熵):</strong> $H(S) = -\\sum_{i=1}^c p_i \\log_2(p_i)$</li>
    <li><strong>Information Gain:</strong> $IG(S, A) = H(S) - \\sum_{v \\in A} \\frac{|S_v|}{|S|} H(S_v)$</li>
    <li><strong>Split Information:</strong> $SplitInfo(A) = -\\sum_{v \\in A} \\frac{|S_v|}{|S|} \\log_2 \\left( \\frac{|S_v|}{|S|} \\right)$</li>
    <li><strong>Gain Ratio (增益率):</strong> $\\frac{IG(S, A)}{SplitInfo(A)}$ （用于克服对多分支属性的偏好）</li>
    <li><strong>Gini Impurity:</strong> $Gini(S) = 1 - \\sum_{i=1}^c p_i^2$ （CART使用）</li>
</ul>
<h2>3. 距离计算与标准化</h2>
<ul>
    <li><strong>Minkowski Distance:</strong> $d(x, y) = \\left( \\sum_{i=1}^n |x_i - y_i|^p \\right)^{\\frac{1}{p}}$ （$p=1$ 为曼哈顿距离，$p=2$ 为欧氏距离，$p \\to \\infty$ 为切比雪夫距离）</li>
    <li><strong>Min-Max Normalization:</strong> $v' = \\frac{v - \\min_A}{\\max_A - \\min_A}$ </li>
    <li><strong>Z-score Standardization:</strong> $v' = \\frac{v - \\mu_A}{\\sigma_A}$</li>
</ul>
<h2>4. AdaBoost 的加权模型推演</h2>
<ul>
    <li><strong>分类器 $m$ 的加权错误率:</strong> $\\epsilon_m = \\sum_{i=1}^N w_i I(y_i \\neq h_m(x_i))$</li>
    <li><strong>分类器权重 $\\alpha_m$:</strong> $\\alpha_m = \\frac{1}{2} \\ln \\left( \\frac{1 - \\epsilon_m}{\\epsilon_m} \\right)$</li>
</ul>
<h2>5. 63.2% Bootstrap 概率推导</h2>
<ul>
    <li><strong>$N$ 次有放回抽样全未命中概率极限:</strong> $\\lim_{N \\to \\infty} \\left(1 - \\frac{1}{N}\\right)^N = \\frac{1}{e} \\approx 36.8\\%$</li>
    <li><strong>单样本被抽中进入训练集的比例:</strong> $1 - 36.8\\% = 63.2\\%$ </li>
</ul>
<h2>6. 关联模式指标</h2>
<ul>
    <li><strong>Support (支持度):</strong> $\\frac{Count(X \\cup Y)}{Total\\ Transactions}$</li>
    <li><strong>Confidence (置信度):</strong> $\\frac{Support(X \\cup Y)}{Support(X)}$</li>
    <li><strong>Lift (提升度):</strong> $\\frac{P(X \\cup Y)}{P(X) \\times P(Y)}$ （>1 表明具有正向促进作用）</li>
</ul>
<h2>7. 聚类分析计算指标</h2>
<ul>
    <li><strong>K-Means 误差平方和 (SSE):</strong> $SSE = \\sum_{i=1}^K \\sum_{x \\in C_i} ||x - \\mu_i||^2$</li>
    <li><strong>Single-Linkage:</strong> $d(A, B) = \\min_{x \\in A, y \\in B} d(x, y)$</li>
    <li><strong>Complete-Linkage:</strong> $d(A, B) = \\max_{x \\in A, y \\in B} d(x, y)$</li>
    <li><strong>OPTICS Core-distance:</strong> 满足 $\\ge MinPts$ 邻居所需的最短半径</li>
    <li><strong>OPTICS Reachability-distance:</strong> $\\max(Core\\_distance(o), d(p, o))$</li>
    <li><strong>Silhouette Coefficient (轮廓系数):</strong> $s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}$</li>
</ul>
"""

m15 = """
<h1>MODULE 15: WEKA 常见算法与策略总结</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>综合提取自全部幻灯片</code></p>
<div class="highlight"><strong>专题说明：</strong>针对算法操作执行流和 WEKA 具体组件命名与机制进行系统归纳。</div>
<h2>1. 分类算法核心流总结 (Classification Flows)</h2>
<ul>
    <li><strong>决策树 J48 (C4.5)：</strong> 采用 Gain Ratio 确定分裂属性，生成决策树后，利用参数 <code>confidenceFactor</code> 触发误差修剪 (后剪枝)，保障泛化能力。</li>
    <li><strong>决策树 REPTree：</strong> 结合方差或信息增益生成决策树，使用预留的 Holdout 验证集，监控泛化错误率并实施 Reduced Error Pruning 后剪枝，有效防止过拟合。</li>
    <li><strong>规则分类 JRip (RIPPER)：</strong> 执行 Separate-and-Conquer 增量提取。运用 MDL 惩罚线限制规则的过度繁杂，提取后即刻剔除已覆盖数据，循环完成全集提取。</li>
    <li><strong>规则提取 PART：</strong> 通过局部递归调用 C4.5 构筑部分决策树，每次抽取最佳叶子路径作为规则。提取完成后舍弃残树，确保获取高度优化的独立规则。</li>
    <li><strong>基准模型 ZeroR：</strong> 忽略所有变量，通过判定多数派 (Majority Class) 直接预测结果。常用于衡量其他算法是否具有增益有效性的基础下限。</li>
    <li><strong>K-NN 近邻 IBk：</strong> 利用内存存储训练数据。测试阶段执行归一化并计算几何距离。支持 <code>distanceWeighting</code> 参数实现倒数加权距离，降低远端噪声干扰。</li>
    <li><strong>随机森林 Random Forest：</strong> Bagging 的进阶方法，在节点分裂时仅在狭窄子集 $K = \\log_2M+1$ 中选取特征，强制引入极大随机性，获得出色的集成多样性和误差平滑效果。</li>
    <li><strong>提升树 AdaBoost：</strong> 强制对错分样本累计提升权重，在逐一接力的错误继承网络中生成专注错题的后续分类器，并利用模型低错率决定权重完成加权判决。</li>
</ul>

<h2>2. 聚类算法机制总结 (Clustering Flows)</h2>
<ul>
    <li><strong>划分聚类 SimpleKMeans：</strong> 使用随机质心引导数据分配并更新重心坐标。通过 Assignment 与 Update 两步循环收拢数据分布，直至 SSE 下降到底部实现聚类固化。</li>
    <li><strong>层次聚类 (Hierarchical AGNES)：</strong> 寻找邻近群组不断整合拼接，采用 Single/Complete 等链路原则构筑自底向上的层级归属树。具备一旦聚并则无法撤销修正的不可逆特性。</li>
    <li><strong>密度聚类 DBSCAN：</strong> 通过指定领域参数 $Eps$ 和密度底线 $MinPts$ 扫描出核心点，依靠连通扩张吸收周边成员或将孤立散点归类为噪声。能够有效规避非凸分布限制与异常点干扰。</li>
</ul>

<h2>3. 模式挖掘与多维引擎 (Frequent Patterns & OLAP)</h2>
<ul>
    <li><strong>Apriori 先验提取：</strong> 在生成 $C_k$ 组合时，依照“所有非空子集必须频繁”的反单调性定律实施严格剪枝，过滤无效组合后再读取数据库计算支持度。</li>
    <li><strong>数据立方体 BUC：</strong> 在下钻切割运算 Cuboid 时，监测其聚合 Count 是否触及冰山阈值限制。一旦跌破，则触发对后续递归降维的所有派生切片进行连坐剪枝阻断，大幅压缩了冗余高维计算开销。</li>
</ul>
"""

# Extract the data directly to a data.js file
data_js_content = f"""const modulesData = {{
  m01: {json.dumps(m01)},
  m02: {json.dumps(m02)},
  m03: {json.dumps(m03)},
  m04: {json.dumps(m04)},
  m05: {json.dumps(m05)},
  m06: {json.dumps(m06)},
  m07: {json.dumps(m07)},
  m08: {json.dumps(m08)},
  m09: {json.dumps(m09)},
  m10: {json.dumps(m10)},
  m11: {json.dumps(m11)},
  m12: {json.dumps(m12)},
  m13: {json.dumps(m13)},
  m14: {json.dumps(m14)},
  m15: {json.dumps(m15)}
}};"""

with open('data.js', 'w', encoding='utf-8') as f:
    f.write(data_js_content)
print("data.js updated with embedded pseudo-code steps in every module.")

# Update index.html to ensure JS logic still remains perfectly decoupled
with open('index.html', 'r', encoding='utf-8') as f:
    html_content = f.read()

# If the <script src="data.js"> isn't present, add it. The file was fixed in previous steps but just in case.
if "data.js" not in html_content:
    start_marker = "const modulesData = {"
    end_marker = "function openModule(modId) {"
    start_idx = html_content.find(start_marker)
    end_idx = html_content.find(end_marker)

    if start_idx != -1 and end_idx != -1:
        new_html = html_content[:start_idx] + "</script>\n<script src=\"data.js\"></script>\n<script>\n" + html_content[end_idx:]
        with open('index.html', 'w', encoding='utf-8') as f:
            f.write(new_html)
        print("index.html fixed with decouple linkage to data.js.")
