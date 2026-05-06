import os
import json

m01 = """
<h1>MODULE 01: 概述与数据预处理</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>Overview.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>KDD流程全貌、数据特征的具体类型、缺失值与冗余特征处理、距离与相似度度量深度分析。</div>
<h2>1. 什么是数据挖掘与知识发现 (KDD)?</h2>
<p>数据挖掘 (Data Mining) 是知识发现 (Knowledge Discovery from Databases, KDD) 过程中最为核心的一环。严格按照课件原图，完整的 KDD 包含以下 <strong>4 个核心步骤</strong>：</p>
<ol>
    <li><strong>Preparation (准备阶段)</strong>
        <ul>
            <li>设定挖掘目标（决定需要学习和发现什么）。</li>
        </ul>
    </li>
    <li><strong>Pre-processing (数据预处理阶段)</strong>
        <ul>
            <li>数据清洗/数据集成 (Data cleaning/integration)：处理噪声、错误以及填补缺失值。</li>
            <li>数据选择/归约/转换/离散化 (Data selection/reduction/transformation/discretization)：筛选相关的样本与变量，构建出用于挖掘的目标数据集。</li>
        </ul>
    </li>
    <li><strong>Data mining (数据挖掘阶段)</strong>
        <ul>
            <li>应用具体的机器学习算法，从预处理好的数据中计算并提取出期望的模式 (patterns)。</li>
        </ul>
    </li>
    <li><strong>Interpretation (解释与部署阶段)</strong>
        <ul>
            <li>如果评估发现模型性能不令人满意，则返回前序步骤进行不断迭代 (Iterate)。</li>
            <li>如果结果准备就绪，则执行部署 (Deploy)：包括生成报告、系统整合以及落地应用。</li>
        </ul>
    </li>
</ol>
<h2>2. 数据集与属性的类型深度剖析</h2>
<p>在 WEKA 的 ARFF 文件格式中，属性类型的设定极其严格。算法的表现高度依赖特征的数据类型：</p>
<ul>
    <li><strong>Nominal / Categorical (标称型 / 类别型)</strong>：如天气（晴、阴、雨）。在 WEKA 中表示为 <code>{Sunny, Overcast, Rainy}</code>。不仅没有大小顺序，也不能做任何代数运算。大部分关联规则算法（如 Apriori）强制要求数据必须全部是标称型。</li>
    <li><strong>Ordinal (序数型)</strong>：如满意度（高、中、低）。有明确顺序，但间隔无意义。算法处理时有时会将其转化为数值，有时转化为标称。</li>
    <li><strong>Numeric / Continuous (数值型 / 连续型)</strong>：包含区间型 (Interval) 和比率型 (Ratio，有绝对零点)。可计算均值和方差，也是 KNN 等计算欧氏距离的必备类型。在 WEKA 中用 <code>numeric</code> 表示。</li>
</ul>

<h2>3. WEKA 数据预处理核心算法流程</h2>
<p><strong>ReplaceMissingValues 过滤器执行流程:</strong></p>
<ol>
    <li>扫描数据集中的每一列特征属性。</li>
    <li>如果该列是 <strong>Numeric (数值型)</strong>，计算该列所有非缺失样本的 <strong>均值 (Mean)</strong>。将所有缺失的 <code>?</code> 替换为均值。</li>
    <li>如果该列是 <strong>Nominal (标称型)</strong>，统计该列各种分类的出现频次，找出 <strong>众数 (Mode)</strong>。将所有缺失的 <code>?</code> 替换为众数。</li>
    <li>如果数据集包含类别标签 (Class Label)，该过滤器可以选择是否针对每个类别的内部独立计算均值/众数，从而让填补更加精确无偏。</li>
</ol>

<h2>4. 距离与相似度度量详解</h2>
<p>数据点之间的接近程度是基于实例学习（如 KNN、K-Means 聚类）的核心：</p>
<ul>
    <li><strong>Minkowski Distance (闵可夫斯基距离)</strong>：广义距离公式 $L_p$ 范数。<br>
        - 当 $p=1$ 时，退化为 <strong>Manhattan (曼哈顿距离)</strong>，计算绝对轴距和：$d(x, y) = \\sum_{i=1}^n |x_i - y_i|$。<br>
        - 当 $p=2$ 时，退化为 <strong>Euclidean (欧氏距离)</strong>，最常用于连续型特征，但对大数值极其敏感：$d(x, y) = \\sqrt{\\sum_{i=1}^n (x_i - y_i)^2}$。<br>
        - 当 $p \\rightarrow \\infty$ 时，为 <strong>Supremum / Chebyshev 距离</strong>（切比雪夫距离）：$d(x, y) = \\max_i |x_i - y_i|$。
    </li>
    <li><strong>Cosine Similarity (余弦相似度)</strong>：常用于高维稀疏数据（如文本向量），测量向量夹角而不是距离，忽略数值的绝对大小，只关注方向的匹配：$\\cos(\\theta) = \\frac{A \\cdot B}{\\|A\\| \\|B\\|}$。</li>
    <li><strong>Jaccard Coefficient (雅卡尔系数)</strong>：专用于非对称的二元布尔属性（比如顾客是否买了某件商品），忽略双 0 匹配（没买A且没买B不代表两者相似），公式为 $J(A, B) = \\frac{|A \\cap B|}{|A \\cup B|}$。</li>
</ul>
<h3>🎯 Mock Exam 经典例题</h3>
<p><strong>【Q1. 简答题】</strong>请严格按照讲义原图定义，列举出完整的 KDD (Knowledge Discovery from Databases) 过程所包含的 4 个核心步骤，并指明哪一个大步骤包含了 Deploy (部署)？</p>
<p><strong>【解答】</strong>: 完整的 KDD 步骤包含 4 大阶段：<br>1. Preparation (准备)<br>2. Pre-processing (预处理)<br>3. Data mining (数据挖掘)<br>4. Interpretation (解释与部署)<br>其中，部署 (Deploy if ready: report, incorporate, apply) 被归类在第四个核心大阶段 <strong>Interpretation (解释)</strong> 当中。</p>
<p><strong>【Q2. 计算题】</strong>给出两个向量 $X = (1, 3)$ 和 $Y = (4, 7)$，请分别计算它们之间的曼哈顿距离、欧氏距离和切比雪夫距离。</p>
<p><strong>【解答】</strong>:<br>
- 曼哈顿距离 ($L_1$): $|1-4| + |3-7| = 3 + 4 = 7$<br>
- 欧氏距离 ($L_2$): $\\sqrt{(1-4)^2 + (3-7)^2} = \\sqrt{9+16} = 5$<br>
- 切比雪夫距离 ($L_\\infty$): $\\max(|1-4|, |3-7|) = \\max(3, 4) = 4$</p>
"""

m02 = """
<h1>MODULE 02: 分类模型性能评估</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>Classification_EV.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>混淆矩阵的全方位推导、Accuracy在极端不平衡下的欺骗性、Recall与Precision的取舍、F-score、ROC曲线与PRC曲线的数学对比。</div>
<h2>1. 混淆矩阵 (Confusion Matrix for Binary Classification)</h2>
<p>评估模型的最直观工具。WEKA 输出结果中必然包含 Confusion Matrix，其中行通常代表真实情况，列代表预测结果。</p>
<ul>
    <li><strong>TP (真正例)</strong>: 癌症患者被诊断为癌症。</li>
    <li><strong>TN (真负例)</strong>: 健康人被诊断为健康。</li>
    <li><strong>FP (假正例 / 误报 False Alarm)</strong>: 健康人被误诊为癌症。在统计学中被称为 <strong>第一类错误 (Type I Error)</strong>。</li>
    <li><strong>FN (假负例 / 漏报 Missed Detection)</strong>: 癌症患者被误诊为健康。在统计学中被称为 <strong>第二类错误 (Type II Error)</strong>。</li>
</ul>
<h2>2. 评价指标及其在业务场景中的权衡</h2>
<ul>
    <li><strong>Accuracy (准确率)</strong> = $\\frac{TP + TN}{TP + TN + FP + FN}$。<strong>致命局限性</strong>：当处理 <strong>Class imbalance problem (类别不平衡问题)</strong> 时完全失效。比如诈骗交易仅占 0.1%，一个把所有交易都判定为正常的弱智模型（ZeroR）能达到 99.9% 准确率，但在业务上毫无意义。</li>
    <li><strong>Precision (精确率 / 查准率)</strong> = $\\frac{TP}{TP + FP}$。代表“报警的案件中，有多少是真的”。警方抓犯人时看重此指标，防止滥抓无辜。</li>
    <li><strong>Recall / Sensitivity / TPR (召回率 / 查全率 / 真正例率)</strong> = $\\frac{TP}{TP + FN}$。代表“所有真正的案件中，我们抓住了多少”。<strong>疾病检测、地震预警极其看重此指标</strong>，因为漏报（FN）的代价远大于误报（FP）。</li>
    <li><strong>Specificity / TNR (特异性 / 真负例率)</strong> = $\\frac{TN}{TN + FP}$。代表真正的健康人中，有多少被正确排除了。</li>
</ul>
<h2>3. 综合评价指标 (F-Measure)</h2>
<p>Precision 和 Recall 总是相互牵制的（提高严格度，Precision 上升，Recall 下降）。为了综合评估，采用调和平均数：<br>
<strong>F1-score</strong> = $2 \\times \\frac{Precision \\times Recall}{Precision + Recall}$。<br>
调和平均数的特点是“木桶效应”，只有当 Precision 和 Recall <strong>双双都很高时</strong>，F1 才会高。</p>
<p><strong>F-beta score</strong> 允许赋予倾向：$F_\\beta = (1 + \\beta^2) \\times \\frac{Precision \\times Recall}{\\beta^2 \\times Precision + Recall}$。当 $\\beta > 1$ 时，更看重 Recall；当 $\\beta < 1$ 时，更看重 Precision。</p>

<h2>4. WEKA 中的基础判别与阈值算法流</h2>
<p><strong>ZeroR 算法执行流程 (分类最差基准):</strong></p>
<ol>
    <li>扫描全库，无视任何所有的描述性特征变量。</li>
    <li>仅仅统计目标类标 (Class Label) 的分布。比如正类出现10次，负类出现90次。</li>
    <li><strong>建立模型</strong>：强制输出类标分布中数量最多的那个类（即 负类）。</li>
    <li><strong>测试阶段</strong>：无论你输入任何新样本的任何特征，永远闭着眼睛输出 负类。这用于证明如果一个高级算法的预测成功率还不如纯猜大多数，那么这个高级算法是负收益。</li>
</ol>
<p><strong>Threshold-moving 曲线画法流程 (ROC/PRC 绘制原理):</strong></p>
<ol>
    <li>测试所有样本，让模型为每个样本输出其被判定为正类的 <strong>概率得分 (Probabilities)</strong>。</li>
    <li>将所有样本按照概率得分从高到低严格排序。</li>
    <li>一开始，将判定阈值设为无穷大。此时所有样本全被猜为负例（TP=0, FP=0）。</li>
    <li>逐渐降低阈值，每次只通过一个新样本将其判定为正例。如果它是真正的正例，TPR (Recall) 爬升一格；如果它是假的正例，FPR 向右爬升一格。</li>
    <li>最终阈值降到0，所有样本全被判定为正，连成整条从 (0,0) 到 (1,1) 的操作特征曲线。</li>
</ol>

<h2>5. ROC 曲线与 PRC 曲线深度对比</h2>
<ul>
    <li><strong>ROC Curve (接收者操作特征曲线)</strong>：纵轴为 TPR (Recall)，横轴为 FPR ($1 - Specificity = \\frac{FP}{FP + TN}$)。完美的模型曲线会直达左上角 $(0, 1)$。<strong>AUC (Area Under Curve)</strong> 等于随机抽取一个正例得分高于负例的概率，AUC 越大越好。</li>
    <li><strong>PRC Curve (Precision-Recall 曲线)</strong>：纵轴为 Precision，横轴为 Recall。</li>
    <li><strong>极限对比考点：ROC vs PRC</strong><br>
        当负例 (TN) 数量极大时（极端不平衡数据），FP 的微小增加会被巨大的 TN 淹没，导致 FPR 看起来依然很低。这会使得 <strong>ROC 曲线呈现出虚假的乐观（依然高高隆起）</strong>。但在此时，Precision 因为分母是 $TP+FP$，没有 TN 的掩护，会断崖式暴跌。<br>
        <strong>结论：面对极端类别不平衡数据，必须使用 PRC 评估模型，ROC 会骗人。</strong></li>
</ul>
<h3>🎯 Mock Exam 经典例题</h3>
<p><strong>【Q1. 矩阵计算题】</strong>在一次针对罕见病的预测中，WEKA 输出了以下混淆矩阵 (行代表实际，列代表预测)：<br>
<pre>
   a   b  <-- classified as
 100  20  | a = 患病 (Positive)
  30 850  | b = 健康 (Negative)
</pre>
请计算 患病类别(Positive) 的 Accuracy, Precision, Recall 和 F1-score。</p>
<p><strong>【解答】</strong>:<br>
- 提取基础值：$TP=100, FN=20, FP=30, TN=850$<br>
- $Accuracy = \\frac{100+850}{100+20+30+850} = \\frac{950}{1000} = 95\\%$<br>
- $Precision = \\frac{100}{100+30} = \\frac{100}{130} \\approx 76.92\\%$<br>
- $Recall = \\frac{100}{100+20} = \\frac{100}{120} \\approx 83.33\\%$<br>
- $F1\\text{-}score = 2 \\times \\frac{0.7692 \\times 0.8333}{0.7692 + 0.8333} \\approx 80\\%$</p>
<p><strong>【Q2. 概念辨析】</strong>在极端不平衡的信用卡欺诈检测数据集中，为什么即使模型非常糟糕，ROC 曲线下的面积 (AUC) 也可能看起来很高？请指出一种更合适的替代评估图表。</p>
<p><strong>【解答】</strong>: 在极端不平衡下，真实负例 (TN) 的数量远大于假正例 (FP)。ROC 的横轴 $FPR = FP / (FP+TN)$。由于 $TN$ 分母极其庞大，导致无论模型犯多少次假阳性错误，$FPR$ 的变化都极其微小，ROC 曲线被强行维持在左上方，呈现出虚假的乐观。<br>应该使用 <strong>Precision-Recall Curve (PRC)</strong>，由于 Precision 的分母是 $TP+FP$，排除了 $TN$ 的掩护，能够最真实地反映模型在少数目标类上的崩溃。</p>
"""

m03 = """
<h1>MODULE 03: Decision Tree Induction (决策树)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>Classification_DT.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>贪心算法的本质缺陷、Information Gain的偏向问题及 Gain Ratio 修正、CART树的 Gini 纯度、连续与离散变量处理、WEKA中决策树算法实现 (J48, REPTree) 的优缺点对比与预/后剪枝机制。</div>
<h2>1. 算法核心哲学：贪心算法 (Greedy Algorithm)</h2>
<p>决策树是一种自上而下、运用 <strong>Divide and Conquer (分而治之)</strong> 思想的算法。它试图在每一步分裂中找到能使子集“纯度”提升最快的一个属性。<br>
<strong>致命局限 (Myopic Problem / 近视问题)</strong>：由于它是一个纯粹的贪心算法，每次分裂仅看当下这一步能否获得最大纯度提升，完全不考虑后续步骤的组合效应。这种“目光短浅”可能会导致最终生成的树陷入局部最优，无法达到结构最紧凑的全局最优状态。</p>
<h2>2. 杂质度量与分裂准则 (Impurity Measures)</h2>
<p>决策树有三个核心派系（ID3/C4.5 派系 与 CART 派系），区别在于分裂标准的计算：</p>
<ul>
    <li><strong>Shannon's Entropy (信息熵)</strong>：计算公式 $H(p) = -\\sum p_i \\log_2(p_i)$。代表系统的混乱程度，类别越均匀，熵越高。
        <ul>
            <li><strong>Information Gain (信息增益, ID3使用)</strong> = 父节点熵 - 子节点加权熵。它倾向于选择能极大降低混乱度的属性。<br>
            <strong>灾难性缺陷 (Bias)</strong>：如果有一个属性叫“身份证号”或“唯一ID”，用它分裂会产生N个分支，每个分支里只有1个样本（纯度100%，子节点熵全部为0）。信息增益会达到最大，算法会认为这是一个极其完美的分割。但这种树毫无泛化能力 (Overfitting)！</li>
            <li><strong>Gain Ratio (增益率, C4.5使用)</strong> = $\\frac{Information\\ Gain}{Split\\ Information}$。Split Info 是这个属性本身分裂所产生的熵：$SplitInfo_A(D) = -\\sum_{j=1}^v \\frac{|D_j|}{|D|} \\log_2 \\left(\\frac{|D_j|}{|D|}\\right)$。由于 ID 分裂产生的底数非常大，Split Info 极大，作为分母直接压低了 Gain Ratio 的值。这完美修正了“偏向多值属性”的问题。</li>
        </ul>
    </li>
    <li><strong>Gini Impurity (基尼不纯度, CART使用)</strong>：公式 $Gini(p) = 1 - \\sum_{i=1}^J p_i^2$。它衡量的是随机抽取两个样本，其标签不一致的概率。Gini 在二元分类下计算比对数快。<strong>倾向特性：Gini 倾向于在数据集中切出大小基本均等的两个分支，而 Entropy 可能会切出一个极大和一个极小的分支。</strong></li>
</ul>

<h2>3. WEKA 决策树大满贯：J48 vs REPTree vs RandomTree 执行全流程</h2>
<p><strong>J48 算法 (对应 C4.5) 核心建树流程 (Divide-and-Conquer):</strong></p>
<ol>
    <li>如果当前节点内所有样本 <strong>同属于一个类别</strong>，直接设为叶子节点，附上该类别标签。</li>
    <li>如果已经 <strong>没有属性可供切分</strong> 或者当前节点的样本数已经少于 <code>minNumObj</code> 的硬性门槛下限，按多数表决法将其封闭为叶节点。</li>
    <li>如果还有特征，遍历计算所有候选属性的 <strong>Gain Ratio (增益率)</strong> 纯度分。</li>
    <li>挑选带来 <strong>最大纯度提升 (Best Split)</strong> 的属性作为分裂节点。</li>
    <li>将数据集彻底打散划分入互不相交的 <strong>子集 (Subsets)</strong>，对每个子集 <strong>递归 (Recursively)</strong> 调用步骤1到4。</li>
    <li>整棵大树构建完毕后，立刻启动 <strong>后剪枝 (Post-pruning)</strong> 阶段。利用参数 <code>confidenceFactor</code> 计算折叠整棵子树带来的误差成本是否比维持现状更小，从而进行大刀阔斧的修剪。</li>
</ol>

<p><strong>REPTree (Reduced Error Pruning Tree) 算法核心流程:</strong></p>
<ol>
    <li>首先，将整个数据集强行切割为 <strong>训练集</strong> 和 <strong>独立验证集 (Holdout set)</strong>。</li>
    <li>使用训练集，基于 Information Gain 或方差快速生长出一棵庞大无比的决策树（速度极快）。</li>
    <li>树建完后，自底向上拿着这棵树去那块 <strong>独立验证集</strong> 上跑跑看。</li>
    <li>如果发现某个底层的分支节点在独立验证集上犯下的错误，比把它全部揉成一团单一的大叶子还要多（即验证集 Error 上升了），证明此分支纯属过度死记硬背训练集的噪点，算法立刻将该分支 <strong>直接砍断 (Reduced Error Pruning)</strong>。</li>
</ol>

<p><strong>RandomTree 算法核心流程:</strong></p>
<ol>
    <li>在每一步准备计算分裂节点时，算法 <strong>绝对不去看所有的特征</strong>。</li>
    <li>它会从所有 $M$ 个特征中，盲盒式地 <strong>随机抽取 $K$ 个特征子集</strong>（通常 $K=\\log_2M+1$）。</li>
    <li>只允许在这随机挑出的 $K$ 个可怜特征里挑一个最好的进行分裂。</li>
    <li>不仅如此，它一路上生根发芽 <strong>完全不执行任何剪枝操作 (No pruning)</strong>，让树长得极深极庞大。</li>
    <li><strong>终极使命：</strong>由于这棵树极度不稳定且方差极大，WEKA 绝不会单独依靠它，而是专门用来扔进 RandomForest (随机森林) 里做成千上万棵树去投票平滑误差的。</li>
</ol>

<h3>🎯 Mock Exam 经典例题</h3>
<p><strong>【Q1. 计算推演题】</strong>在一个包含 10 条训练样本的节点中，有 6 个是 "Yes"，4 个是 "No"。若我们利用属性 <code>Wind</code> (具有两个分支：Strong 和 Weak) 对其分裂，其中 Strong 分支掉入 2 个 "Yes" 和 3 个 "No"；Weak 分支掉入 4 个 "Yes" 和 1 个 "No"。请写出该次分裂 Information Gain 的计算公式（无需算出最终对数值）。</p>
<p><strong>【解答】</strong>:<br>
- 首先计算父节点的初始信息熵：$H(S) = -\\left(\\frac{6}{10} \\log_2 \\frac{6}{10} + \\frac{4}{10} \\log_2 \\frac{4}{10}\\right)$<br>
- 然后计算 Strong 分支（共5条）的局部熵：$H(Strong) = -\\left(\\frac{2}{5} \\log_2 \\frac{2}{5} + \\frac{3}{5} \\log_2 \\frac{3}{5}\\right)$<br>
- 接着计算 Weak 分支（共5条）的局部熵：$H(Weak) = -\\left(\\frac{4}{5} \\log_2 \\frac{4}{5} + \\frac{1}{5} \\log_2 \\frac{1}{5}\\right)$<br>
- 最终的 Information Gain 为：$IG = H(S) - \\left( \\frac{5}{10} H(Strong) + \\frac{5}{10} H(Weak) \\right)$</p>
<p><strong>【Q2. 算法填空题】</strong>在 WEKA 中，如果我希望建立一棵极度简单的决策树，叶子节点的样本不能低于 20 个，且剪枝要非常狠，我应该使用 ____ 算法，将参数 <code>minNumObj</code> 设为 ____，并将参数 <code>confidenceFactor</code> 设得更 ____ (大/小)？</p>
<p><strong>【解答】</strong>: 应该使用 <code>J48</code> 算法，将参数 <code>minNumObj</code> 设为 <code>20</code>，并将参数 <code>confidenceFactor</code> 设得更 <code>小</code>（因为 $C$ 越小，代表算法能容忍的误差波动空间越严苛，就会剪去更多分支）。</p>
"""

m04 = """
<h1>MODULE 04: K-Nearest Neighbor (KNN)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>Classification_NN.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>懒惰学习 (Lazy Learning) 的深刻内涵、Voronoi 图决策边界、距离度量细节、未归一化导致的权重灾难、以及 WEKA IBk 算法的原理。</div>
<h2>1. 懒惰学习 (Lazy Learning) 与 急切学习 (Eager Learning) 的对立</h2>
<p>绝大多数算法（如决策树、神经网络）属于 <strong>Eager Learning</strong>。它们在面对训练集时，立刻耗费巨大的算力，试图归纳出一个普适的全局规律或数学模型，从而把训练数据本身丢弃。预测时极快。</p>
<p>KNN (K-Nearest Neighbor) 则是一种 <strong>Lazy Learner (懒惰学习器 / 基于实例的学习)</strong>。在训练阶段，它的计算复杂度几乎为零，它仅仅是将所有的训练样本毫无保留地装进内存。所有的繁重计算全部被推迟到了 <strong>预测阶段 (Testing phase)</strong>：面对每一个新样本，它必须遍历庞大的内存，计算新点与历史上所有存储点的几何距离。</p>
<h2>2. Voronoi Diagram 与决策边界</h2>
<p>KNN 没有任何显式的函数边界。在二维空间中，如果你设定 $K=1$，空间将被训练样本硬生生地分割成无数个多边形格子。每个格子由其中唯一的一个训练样本统治，任意落入格子内的新样本，其距离该样本都是最近的。由样本点之间的垂直平分线构成的这种网格图，在计算几何中被称为 <strong>Voronoi Diagram (泰森多边形 / 沃罗诺伊图)</strong>。</p>
<h2>3. 维度度量与归一化 (Normalization) 的生杀大权</h2>
<p>在使用欧几里得距离 $D = \\sqrt{\\sum (x_i - y_i)^2}$ 计算时，存在一个 <strong>极其致命的陷阱</strong>：<br>
如果属性 A 代表“年收入”(范围 10,000 ~ 100,000)，属性 B 代表“孩子个数”(范围 0 ~ 5)。在直接计算距离差时，年收入的平方差达到了几十亿，而孩子个数的平方差最多只有几十。这意味着：<strong>未经归一化的 KNN，单位量纲大的特征将彻底独裁距离的判定，数值小的特征直接丧失发言权！</strong></p>
<p><strong>解决方案：</strong></p>
<ul>
    <li><strong>Min-Max Normalization (最小-最大规范化)</strong>：将数据线性映射至 $[0, 1]$ 区间。公式：$v' = \\frac{v - \\min}{\\max - \\min}$。极度容易受到离群点 (outliers) 干扰（一个异常极大值会把正常值全压扁在0附近）。</li>
    <li><strong>Z-score Standardization (标准正态规范化)</strong>：公式 $v' = \\frac{v - \\mu}{\\sigma}$。处理离群点时远比 Min-Max 稳定。</li>
</ul>
<h2>4. 超参数 $K$ 值的极性影响</h2>
<ul>
    <li><strong>$K$ 太小 (如 $K=1$)</strong>：决策边界支离破碎，模型会去拟合周围每一个孤立的噪点 (Noise)。导致极高复杂度的 <strong>过拟合 (Overfitting)</strong>。</li>
    <li><strong>$K$ 极大 (如 $K=N$)</strong>：算法完全退化，对任何人都会预测为样本集中占绝对多数的那个类别 (Majority Class)，这导致 <strong>欠拟合 (Underfitting)</strong>。</li>
</ul>

<h2>5. WEKA 中的 KNN 算法流：IBk (Instance-Based learning)</h2>
<p>在 WEKA 中，KNN 对应的核心落地算法是 <strong>IBk</strong> (位于 <code>lazy</code> 分类目录下)。由于是懒惰学习，它的执行流截然不同：</p>
<p><strong>IBk 算法核心测试与预测流程:</strong></p>
<ol>
    <li><strong>[Training 阶段]：</strong>算法耗时 $O(1)$，纯粹只是在底层的 <code>LinearNNSearch</code> 或高级的 <code>KDTree</code>/<code>BallTree</code> 空间数据结构里分配一块内存，把所有带有类标的训练数据粗暴地塞进去保存，不建立任何概率模型。</li>
    <li><strong>[Testing 阶段发力]：</strong>一旦收到一个新的测试实例 $X$，IBk 首先强制调用底层归一化机制，使得 $X$ 身上所有的属性和历史数据的刻度彻底对齐拉平。</li>
    <li><strong>[Scan & Distance]：</strong>在整个高维空间历史库中扫描，利用欧氏距离计算 $X$ 与内存中每一个点的几何距离。</li>
    <li>挑选出距离 $X$ 最近的排行榜前 $K$ 位历史邻居（由参数 <code>k</code> 决定）。</li>
    <li><strong>[Weighting 加权裁决]：</strong>检查 <code>distanceWeighting</code> 参数。如果不加权，前 $K$ 个人每人一票民主选举；如果启用了 <strong>Inverse Distance (距离倒数加权)</strong>，则给贴得最近的邻居施加极大的权重倍数 $w_i = \\frac{1}{d(x, x_i)}$，从而狠狠打压边缘偶然噪点的投票权。最后根据得票最高者定下 $X$ 的最终类标。</li>
</ol>

<h3>🎯 Mock Exam 经典例题</h3>
<p><strong>【Q1. 归一化与距离计算题】</strong>有三个样本拥有两个特征(Salary, Age)：$A(50000, 30)$，$B(60000, 40)$，新样本 $X(55000, 35)$。已知 Salary 的历史极小极大值是 $[10000, 110000]$，Age的极小极大值是 $[20, 70]$。请使用 Min-Max 归一化处理所有数据，并计算新样本 $X$ 距离 $A$ 和 $B$ 的欧氏距离。</p>
<p><strong>【解答】</strong>:<br>
- $Salary$ 归一化分母 = $110000-10000 = 100000$。$A_{sal}'=0.4, B_{sal}'=0.5, X_{sal}'=0.45$。<br>
- $Age$ 归一化分母 = $70-20 = 50$。$A_{age}'=\\frac{10}{50}=0.2, B_{age}'=\\frac{20}{50}=0.4, X_{age}'=\\frac{15}{50}=0.3$。<br>
- 归一化后坐标：$A(0.4, 0.2)$， $B(0.5, 0.4)$， $X(0.45, 0.3)$。<br>
- 距离 $X$ 到 $A$: $\\sqrt{(0.45-0.4)^2 + (0.3-0.2)^2} = \\sqrt{0.0025 + 0.01} = \\sqrt{0.0125} = 0.1118$<br>
- 距离 $X$ 到 $B$: $\\sqrt{(0.45-0.5)^2 + (0.3-0.4)^2} = \\sqrt{0.0025 + 0.01} = \\sqrt{0.0125} = 0.1118$。<br>所以 $X$ 到 $A$ 和 $B$ 的距离严格相等。</p>
<p><strong>【Q2. 概念辨析】</strong>如果 KNN 模型发生了严重的过拟合 (Overfitting)，即在测试集上表现极差，我们通常应该增大还是减小 $K$ 值？为什么？</p>
<p><strong>【解答】</strong>: 应该 <strong>增大 $K$ 值</strong>。因为过拟合意味着模型对局部的微小噪点和极端离群点作出了过分复杂的反应（$K$ 极小时）。增大 $K$ 会让更多的邻居参与投票，从而平滑掉个别噪点的干扰，让决策边界变得更平滑、稳定。</p>
"""

m05 = """
<h1>MODULE 05: Rule-Based Classification (基于规则的分类)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>Classification_RB.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>规则提取的极高解释性、Sequential Covering (分离并征服) 与分而治之的本质区别、RIPPER防止过拟合的 MDL 原则、WEKA中 JRip、PART的运作细节。</div>
<h2>1. 基于规则分类的不可替代性</h2>
<p>即使在神经网络(Black-box)大行其道的今天，Rule-Based 系统在金融风控、医疗拒保系统、法律定罪等领域依然是王者。因为一组精简的 <strong>IF-THEN 规则</strong> 具备人类可以直接阅读、审计、并强行进行局部删改的特点（即 <strong>Modular 模块化</strong> 与 <strong>Interpretable 解释性</strong>）。一棵庞大的决策树如果改动一个上层节点，整棵树都必须重构，而规则集可以直接禁用某一条不良规则。</p>
<h2>2. 规则的覆盖冲突 (Conflict Resolution)</h2>
<p>如果新来一个用户，他同时触发了 Rule 1 (判死刑) 和 Rule 2 (判无罪)，怎么办？</p>
<ul>
    <li><strong>Size ordering</strong>：优先相信触发条件苛刻（前件更长更具体）的规则。</li>
    <li><strong>Class-based ordering</strong>：按类别的罕见性排序（比如优先判别为少数派欺诈类别）。</li>
    <li><strong>Rule-based ordering (Decision List)</strong>：将规则严格按优先级排成一个长长的 List。测试实例顺着 List 往下走，触发了哪条就立刻输出结果并停止验证后面的规则。WEKA 中的 JRip 产出的就是这种 Decision List。</li>
</ul>

<h2>3. WEKA 纯血规则引擎：JRip 与 PART 核心算法流深度剖析</h2>
<p>由于庞大的树结构一旦提取规则会导致重重冗余，规则流派选择了最血腥的 <strong>Separate-and-Conquer (分离并征服 / Sequential Covering)</strong> 工作流。</p>
<p><strong>JRip (对应的原算法为 RIPPER) 的增量学习流:</strong></p>
<ol>
    <li>创建一个完全空白的黑板作为最终规则集。</li>
    <li><strong>[Learn-One-Rule (学完一条)]</strong> 在当前场上所有还存活着的样本上，通过 <strong>FOIL 信息增益</strong> 不断添加 <code>AND</code> 判定条件，组装出一条针对某个目标类覆盖率最好、最完美的单薄规则。</li>
    <li><strong>[防爆死阶段 (MDL Stopping)]</strong>：在拼装这唯一一条规则时，时刻监控模型体积。一旦发现加上新条件带来的识别正确率，弥补不了它导致模型位宽膨胀超过 <strong>64 bits 最小描述长度 (MDL)</strong> 的惩罚，算法立刻叫停，这条规则正式出炉。</li>
    <li><strong>[Remove-Covered (拔草除根)]</strong>：去训练集中，将刚才这条好规则能够覆盖抓取出来的所有样本，<strong>彻彻底底地物理删除隔离</strong>！</li>
    <li>回到步骤2，只盯着那些依然没有被提取规律的顽固样本，继续长出一条新规则。直到场上没有任何样本剩下。</li>
    <li>最终进入 <strong>Global Optimization (全局大修剪)</strong> 阶段，将各个单点生长的规则整合去重，生成有序的 Decision List。</li>
</ol>

<p><strong>PART 算法流 (借刀杀人的暴力美学):</strong></p>
<ol>
    <li>在当下的全量数据上不慢吞吞找规则，而是直接调用底层的 C4.5 (J48) 代码，在原地拔地而起一棵庞大复杂的 <strong>局部部分决策树 (Partial Decision Tree)</strong>。</li>
    <li>在这棵树上的成百上千个分岔树叶中，仅仅寻找那一条 <strong>纯度最纯、覆盖样本面积最大</strong> 的那一条黄金路径。将其直接扯断下来，翻译成 <code>IF...THEN...</code> 规则加入集合。</li>
    <li>紧接着做出令人发指的动作：<strong>立刻把这棵刚建好的大树丢进垃圾桶抛弃 (Discard the tree)！</strong></li>
    <li>将这条黄金规则覆盖到的实例从数据集中删除。回到步骤1再次种树。这种“杀鸡用牛刀”的方式虽然算力损耗恐怖，但提取出的规则纯净度惊人。</li>
</ol>

<h3>🎯 Mock Exam 经典例题</h3>
<p><strong>【Q1. 算法执行流程考点】</strong>请阐述 JRip（WEKA中的 RIPPER 算法）是依靠什么底层的停止机制来防止提取出来的规则集过于庞杂、死记硬背而导致过拟合的？</p>
<p><strong>【解答】</strong>: JRip 依靠 <strong>MDL (Minimum Description Length, 最小描述长度)</strong> 原则作为它的核心停止机制 (Stopping criterion)。在执行 Sequential Covering (逐步增加新规则) 或扩充某条规则前件时，算法会实时监控这套规则在理论上的“位宽/信息体积”。如果最新加入的一条规则虽然略微提高了训练集的纯净度，但它带来的模型长度扩充超过了容忍的惩罚阈值（通常设定为 64 bits），系统就会认定得不偿失，立刻叫停该规则的生长以保持模型的泛化能力。</p>
<p><strong>【Q2. 概念对比】</strong>PART 和 JRip 虽然都是用来挖掘独立规则集的，请问它们在生成“最好的一条规则”时的策略有何本质区别？</p>
<p><strong>【解答】</strong>: JRip 采用的是增量生长策略，每次挑一个最优特征（通过 FOIL 信息增益）直接接上 <code>AND</code> 条件；而 PART 的做法极度暴力——它利用当前所有未被覆盖的实例，直接调用类似 C4.5 的建树算法去强制生长出一棵“残缺版的局部决策树 (Partial Tree)”，然后挑这棵树上看起来覆盖面积最庞大、纯度最好的叶子节点路径强行拔下来作为提取到的第一条规则，然后立刻把剩下的树丢进垃圾桶。这种“局部造树”赋予了它极高的准确性，但计算损耗极大。</p>
"""

m06 = """
<h1>MODULE 06: Ensemble Methods (集成学习)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>EnsembleMethods.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>Bias-Variance 理论下的偏差方差权衡、Bagging的Bootstrap抽样平滑方差机制、Random Forest特征随机化、Boosting(AdaBoost)的加权纠错降低偏差机制。</div>
<h2>1. 为什么要使用集成方法？(Ensemble Philosophy)</h2>
<p>单一的分类器往往具有不可逾越的瓶颈：浅层决策树虽然稳健但 <strong>偏差 (Bias)</strong> 极高，只能划出四四方方的决策块（欠拟合）；深层决策树虽然拟合力强，但极其容易被局部噪声诱导，<strong>方差 (Variance)</strong> 极高，泛化一塌糊涂。<br>
<strong>Ensemble (集成)</strong> 的哲学是：将多个相互独立且具有 <strong>多样性 (Diverse)</strong> 的弱基分类器结合起来，能够平滑噪声降低方差，或串联纠错降低偏差，从而打破单一算法的性能天花板。这也是所有 Kaggle 竞赛中的屠榜利器。</p>

<h2>2. WEKA 集成神阵：四大融合算法执行流</h2>
<p><strong>Bagging (Bootstrap Aggregation 装袋法) 算法流：解决高方差 (Reduce Variance)</strong></p>
<ol>
    <li>设原始数据库拥有 $N$ 条样本。启动 $M$ 次平行的随机线程。</li>
    <li><strong>[Bootstrap 抽样]</strong>：每个线程在数据库里进行 <strong>有放回的抽取</strong> $N$ 次，拼凑出一块新数据集（注：约 36.8% 的原数据永远没被抽中，形成 OOB 数据）。</li>
    <li>每个线程拿着自己手上的新数据，互相不知道对方存在，独立地疯狂生长出一棵不受限制的高复杂基础分类器（如深树、KNN）。</li>
    <li><strong>[Voting 民主融合]</strong>：新样本降临，这 $M$ 棵极易过拟合的基分类器全体发言。如果是分类则采用 <strong>多数投票 (Majority Voting)</strong>，如果是回归则全盘 <strong>取平均 (Average)</strong>。此时奇迹发生：个别极端噪点引发的错误在平均下彻底被浩荡票数淹没，全局方差产生指数级崩塌。</li>
</ol>

<p><strong>Random Forest (随机森林) 算法流：解决 Bagging 树同质化的极致升华</strong></p>
<ol>
    <li>完全继承上文 Bagging 的 Bootstrap 随机抽人法则。</li>
    <li>但是！当每棵树进入生长阶段，正准备找那个带来最大纯度 Gain 的特征时，算法 <strong>蒙住它的眼睛</strong>，不许它看全场所有的特征列！</li>
    <li>只允许它从所有特征中随机摸取一个小碎片子集（数量通常为 $\\log_2(总特征数)+1$）。</li>
    <li>树只能在这个随机挑选出来的残缺特征群中，勉为其难挑选出最好的一个去劈出树根。这导致原本会长得一模一样的森林，被强行分散出了极富多变、千奇百怪的独特变异物种，为最终集成平滑带来了 <strong>无与伦比的多样性 (Diversity)</strong>。</li>
</ol>

<p><strong>AdaBoost (Adaptive Boosting) 算法流：解决高偏差 (Reduce Bias) 的专科庸医拯救者</strong></p>
<ol>
    <li>绝对不允许并行！它必须将一群只会画一条线（如单层决策树桩 Decision Stump）的低智分类器排成一个单线纵队。一开始，所有的样本都被分配极其平等的初始权重。</li>
    <li><strong>[ Sequential Learning 逐个传功]</strong>：第一棵弱智树开始学习，由于它很笨，它分错了一堆离群高难样本。</li>
    <li>系统立刻震怒：把这些被分错的考题，其 <strong>难度权重强行提升数倍 (Increase weight for misclassified instances)</strong>。</li>
    <li>轮到第二棵弱智树登场，它看着这些权重冲天的高难考题，被迫拼死专注地去迎合纠正这批错题的命运规律。接着它也分错了部分，于是这部分继续加权传给第三棵树。这叫 <strong>专注弥补前任过错机制</strong>。</li>
    <li><strong>[Weighted Voting 独裁投票]</strong>：测试来临，这排纵队全员投票，但不是人人一票！系统对在当时训练时 <strong>整体加权错误率低</strong> 的优秀生赋予极度夸张的决策发言权 ($\\alpha_m = \\frac{1}{2} \\ln \\left(\\frac{1-\\epsilon}{\\epsilon}\\right)$)，而总是出错的树发言形同放屁。最终用这种强硬的专治力量扭曲出极高拟合深度的非线性边界。</li>
</ol>

<p><strong>Stacking (高阶套娃) 算法流:</strong></p>
<ol>
    <li>不再用同一种算法，你可以第一层摆上完全不搭嘎的三种魔法：一个 J48 树、一个 IBk 和一个 NaiveBayes 贝叶斯。</li>
    <li>给同一个测试样本喂给他们，三个人分别大喊出各自的判定胜率（如 0.9, 0.2, 0.7）。</li>
    <li><strong>最恐怖的一步：将这三个数字直接打包装袋，作为 3 个全新的输入特征 (New Features)</strong>。</li>
    <li>把它们喂给位高权重的最高统帅——<strong>元分类器 (Meta Classifier)</strong>，让它依靠机器学习直接吃透到底该听信谁的谎言与真话。</li>
</ol>

<h3>🎯 Mock Exam 经典例题</h3>
<p><strong>【Q1. 加权公式推演题】</strong>在 AdaBoost 中，如果第 $m$ 棵基分类器（Decision Stump）的加权错误率 $\\epsilon_m$ 被算出来等于 0.2，请问分配给这棵树的发言权权重 $\\alpha_m$ 是多少？（仅需写出计算公式的代入过程）</p>
<p><strong>【解答】</strong>:<br>
在 AdaBoost 中，单棵树的发言权公式为：$\\alpha_m = \\frac{1}{2} \\ln \\left( \\frac{1 - \\epsilon_m}{\\epsilon_m} \\right)$。<br>
代入 $\\epsilon_m = 0.2$，得到 $\\alpha_m = \\frac{1}{2} \\ln \\left( \\frac{1 - 0.2}{0.2} \\right) = \\frac{1}{2} \\ln (4)$。<br>
这意味着因为它的错误率很低，它将在最终投票环节享有很大的话语权权重！</p>
<p><strong>【Q2. 概念分析】</strong>为了解决单一极其深层的决策树产生的过度碎片化预测与严重过拟合问题，你更倾向于选择 AdaBoost 还是 Random Forest 算法进行改进？为什么？</p>
<p><strong>【解答】</strong>: 绝对应该选择 <strong>Random Forest (随机森林) / Bagging 派系</strong>。<br>
极其深层的决策树是典型的**低偏差、极高方差**（Low Bias, High Variance）模型。Bagging 的核心目标就是通过 Bootstrap 多重独立采样与强制随机选取特征，让上百棵不同的高方差深树互相投票，通过平均效应瞬间抹平随机波动的方差 (Reduce variance) 解决过拟合。<br>相反，AdaBoost 的设计初衷是**降低偏差**（Reduce bias），它是用来拯救像单层浅树（Decision stump）这类根本拟合不出复杂边界的“高偏差”弱分类器的。把极深树放进 AdaBoost 里不但无效，反而会进一步放大对局部噪点的串行极度过拟合。</p>
"""

m07 = """
<h1>MODULE 07: Evaluation & The Problem of Overfitting (过拟合与模型评估)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>Evaluation.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>深刻理解泛化与过拟合的本质对立，经验风险最小化(ERM)理论，以及交叉验证、63.2 Bootstrap的数据切分数学原理与应用场景。</div>
<h2>1. 问题的本质：Generalization (泛化) vs Overfitting (过拟合)</h2>
<p>我们训练模型的终极目的，绝对不是让模型去记住历史数据，而是为了应对 <strong>未来全新的、未知的数据 (Unseen / Future data)</strong>。这就叫做模型的 <strong>Generalization (泛化能力)</strong>。<br>
<strong>Overfitting (过拟合)</strong> 指的是：当模型拥有极高的复杂度（参数过多、树极深）时，它能够像背诵课文一样，把训练集 (Fitted data) 中的特异性噪声、巧合、甚至是离群点完美地记下来，导致训练准确率高达 100%。但只要遭遇没有见过的新样本，它的性能会发生断崖式的崩溃。</p>
<h2>2. 经验风险最小化 (Empirical Risk Minimization, ERM) 的两难</h2>
<p>理想情况下，我们应该基于自然界万物的“真实联合概率分布”去寻找能够让“期望风险 (Expected Risk)”最小的分类器。<br>
但在现实世界中，真实的整体分布是不可知的，我们手中只有几百上千条的 <strong>有限样本抽样</strong>。因此所有机器学习算法退而求其次，只能去最小化这些已知样本上的错误率，即 <strong>经验风险最小化 (ERM)</strong>。<br>
这就是过拟合诞生的根本原因：由于 ERM 是在迎合那部分有限的样本均值，如果你的算法过于贪婪或缺乏正则化约束，它就会将这部分有偏差的局部样本特征当作宇宙真理去拟合。</p>
<h2>3. WEKA 数据分割验证引擎流分析</h2>
<p>既然不能用训练集的准确率骗自己，在 WEKA 中提供了多套引擎流程来隔绝测试数据：</p>
<p><strong>Cross-Validation (交叉验证 / K-Fold CV) 核心调度流:</strong></p>
<ol>
    <li>为了绝对公平且榨干所有的数据集潜力，如果设定 10-Fold CV，WEKA 会将整个盘子里的原始数据严格切割为 <strong>10 个不互斥重叠的等额块 (Folds)</strong>。</li>
    <li>并且在切割时，它默认激活 <strong>Stratification (分层机制)</strong>：即死死卡住每个块中的类别分布。如果全体里绝症占 1%，那么这切出来的 10 个块里每一个的绝症都只能精准占到 1%，避免了全员正常人的废块。</li>
    <li>执行长达 10 轮的疯狂大风车旋转。在第 $i$ 轮时，抽出第 $i$ 个块单独封存为不给模型看到的 Test Set，然后把剩下的整整 9 块缝合在一起拿去给分类器喂料训练。</li>
    <li>10轮完毕后，每一块都被拿来单独测过一次。将这 10 个独立测试结果取数学平均值，从而获得了最抗压、方差最低的模型真实泛化分数。</li>
</ol>

<p><strong>63.2 Bootstrap 评估引擎的极限求生法则:</strong></p>
<ol>
    <li>如果数据集小到连 10-Fold CV 都嫌浪费数据怎么破？WEKA 此时激活 Bootstrap。</li>
    <li>它通过极度无赖的 <strong>有放回随机抽样 (Sampling with replacement)</strong> 狂抽 $N$ 次。</li>
    <li><strong>极其高频考点推导</strong>：在有放回抽样中，某一条特定样本在 1 次抽取中 <strong>没有被抽中</strong> 的概率是 $\\left(1 - \\frac{1}{N}\\right)$。那么在 $N$ 次连抽中，它始终没有被抽进训练集的概率是 $\\left(1 - \\frac{1}{N}\\right)^N$。根据极限，当 $N \\to \\infty$ 时，概率趋近于 $\\frac{1}{e} \\approx 0.368$。</li>
    <li>结果就是，被系统生成的“训练池”虽然体积是 100%，但它里面全是重影翻新的多胞胎，只包含大概 <strong>63.2% 的不重复纯净原始原核样本</strong>。</li>
    <li>而刚才被公式推导遗落在荒野的那 <strong>36.8% 的 Out-of-Bag (OOB) 绝版样本</strong>，由于根本没沾染过训练集的气息，被顺理成章地拿来直接当 Test Set 考验模型。</li>
</ol>
<h3>🎯 Mock Exam 经典例题</h3>
<p><strong>【Q1. 计算推演题】</strong>某医疗研究机构仅收集到了 100 份罕见病样本进行数据挖掘。如果采用传统 Holdout，测试集过小会导致评估严重失真，于是他们决定采用 63.2% Bootstrap 策略。请问他们必须在原库中执行多少次“有放回抽样”来组建训练集？在这些生成的训练集中，预期会有多少条其实是互不相同的独特样本？剩下的那部分原本没被抽中的孤立样本被拿去做什么用？</p>
<p><strong>【解答】</strong>:<br>
- 必须执行与原样本数相等次数的有放回抽样，即 <strong>100次</strong>。<br>
- 根据极限公式，在经过大量的有放回抽样后，预期有约 $63.2\\%$ 的独特样本进入训练集。所以训练集中大约包含 <strong>63 条互不相同的独立样本</strong>（其余 37 条是由这些独特样本随机重复而拼凑成满 100 条的池子）。<br>
- 剩下的那部分（约 36.8%）始终没有任何一次被抽中、未被污染的原生数据被称为 <strong>Out-of-Bag (OOB) 样本</strong>，它们将被直接充当绝佳的无偏 <strong>Test Set (测试集)</strong>，用来评估该次训练出来的模型性能！</p>
"""

m08 = """
<h1>MODULE 08: Frequent Pattern Analysis (Apriori Algorithm)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>FrequentPatternAnalysis_AP.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>从事务分析到支持度置信度的数学逻辑，Apriori反单调性与候选生成-剪枝闭环，以及极大与闭合模式的存储压缩，WEKA中关联规则的运行指标。</div>
<h2>1. 市场篮子分析的背景 (Market Basket Analysis)</h2>
<p>这是无监督学习中极为经典的模式挖掘。目标是在不依赖任何先验标签的情况下，通过遍历茫茫的顾客交易记录数据库（Transaction Databases），挖掘出那些 <strong>频繁且同时出现</strong> 的物品组合（例如经典的“买尿布的人倾向于同时买啤酒”）。这直接服务于零售业的货架排列优化、电子商务的交叉销售推荐等。</p>
<ul>
    <li><strong>Support (支持度)</strong>：某项集 $X$ 在所有交易记录中出现的概率。衡量某物是否真的是热门组合：$Support(X) = \\frac{Count(X)}{Total\\ Transactions}$。</li>
    <li><strong>Confidence (置信度)</strong>：条件概率 $P(Y|X)$，即在购买了 $X$ 的交易中，同时包含 $Y$ 的比例。公式为 $Confidence(X \\Rightarrow Y) = \\frac{Support(X \\cup Y)}{Support(X)}$。用来生成类似 <code>IF X THEN Y</code> 的关联规则。</li>
</ul>
<h2>2. Apriori 核心灵魂：反单调性 (Anti-monotone)</h2>
<p>如果商店有 1000 件商品，其可能的所有组合多达 $2^{1000}$ 种。去数据库中挨个查询这些组合出现的次数是不可能的。为了打破组合爆炸 (Combinatorial explosion)，引入了 <strong>Apriori 先验性质 (Downward Closure / 反单调性)</strong>：<br>
<strong>核心真理：一个频繁项集的所有非空子集也必然是频繁的！</strong><br>
<strong>极其凶残的剪枝推论</strong>：如果你去数据库查了一遍，发现连单独买 {啤酒} 的交易都达不到最小支持度阈值（它不是频繁1项集），那么无论你往里面加什么，{啤酒, 尿布}、{啤酒, 牛奶} <strong>绝不可能是频繁的</strong>。这意味着后续庞大的包含啤酒的组合树可以直接被彻底整枝抛弃 (Pruned)！</p>

<h2>3. WEKA 中 Apriori 关联引擎的核心闭环提取流</h2>
<p>Apriori 是一个以暴制暴、逐层攀升迭代的算法引擎，它要挖掘出所有高于门槛底线的项集：</p>
<ol>
    <li><strong>[Initial Scan] 扫描发源</strong>：第一次大范围盘点整个数据库的流水清单，统计出所有单一商品出现的次数，冷酷无情地用 <code>Min_Support</code> 参数将达不到频率基线的货品清盘踢出。留下的胜者被记录在第一代 <strong>频繁 1-项集 ($L_1$)</strong> 名册中。</li>
    <li><strong>[Join 内部联姻]</strong>：此时不再碰数据库。让系统刚拿到的上一代频繁集合 $L_{k-1}$ 内部开始互相两两相亲拼接，通过自我组合生成出比它高一维度的 <strong>候选拼接集 $C_k$</strong>。</li>
    <li><strong>[Prune 极端大屠杀剪枝]</strong>：这正是该引擎性能神话的阵眼。针对 $C_k$ 名单上的每一条候选组合条目，算法会让它分裂出自己所有的降维 $(k-1)$ 级分身子集。如果系统在这个分身里 <strong>只要发现哪怕有一只眼生的小鬼</strong> 根本不存在于上一代的贵族名单 $L_{k-1}$ 里面，那么根据伟大的反单调性铁律，这个候选条目全家立刻被直接在 $C_k$ 中 <strong>彻底删除抹杀 (Prune)</strong>！</li>
    <li><strong>[Validation 二次过滤]</strong>：只有经历了大屠杀没被波及幸存下来的纯血 $C_k$ 组合，系统才舍得浪费读写性能，带着它们 <strong>第二次钻进原始数据库</strong> 去扫表数数，看看它们到底齐不齐，满足门槛的，封为当代的真正大统领 <strong>$L_k$</strong>。</li>
    <li>只要 $L_k$ 还没绝种空城，升维令 $k = k+1$，无休止地跳回第二步 <code>Join</code> 再次厮杀拼接。</li>
</ol>

<h2>4. 高维项集的存储灾难与压缩技术 (Maximal & Closed)</h2>
<p>如果找到一个极其长的频繁模式（包含 100 件商品），那么它的子集就有极大数量（$2^{100}$），这些子集按照定理全都是频繁的。内存会直接爆炸。为了压缩空间，我们只保存最具代表性的项集：</p>
<ul>
    <li><strong>极大频繁项集 (Maximal frequent itemsets)</strong>：自身频繁，但没有任何一个包含了它的真正超集 (proper supersets) 是频繁的。保存它，就等同于暗示其内部所有子集都是频繁的。</li>
    <li><strong>闭合模式 (Closed Patterns)</strong>：自身频繁，并且没有任何一个包含了它的真正超集与它拥有 <strong>完全一样相同的出现计数 (count)</strong>。这不仅仅压缩了边界，还完美保留了所有子集精确的计数支持度信息（因为子集的计数不可能低于它，如果不等于它，肯定比它大，可以另行推算）。</li>
</ul>
<h2>5. 规则评估的黑洞：Confidence 的误导与 Lift</h2>
<p>用高 Confidence 来盲目得出关联规则会产生著名的误导问题。<br>
假设：规则 <code>玩游戏 -> 买iPhone</code> 置信度有 90%，看似极其相关。<br>
但如果在全局数据库中，不论你玩不玩游戏，人类总群体买 iPhone 的概率本身就是 95%！<br>
也就是说，玩游戏不仅没有促进你买 iPhone，反而拉低了你买 iPhone 的概率（从 95% 降到 90%）。此时的高置信度是一种虚假的因果错觉。<br>
因此在 WEKA 的 Apriori 实现中，除了 Support 和 Confidence，还会提供 <strong>Lift (提升度)</strong> 作为补充：$Lift(X, Y) = \\frac{P(X \\cup Y)}{P(X) \\times P(Y)}$。只有 $Lift > 1$，才代表 X 对 Y 是具有真实促进作用的强正相关关系。</p>

<h3>🎯 Mock Exam 经典例题</h3>
<p><strong>【Q1. 置信度与提升度公式连发】</strong>在某个拥有 100 条流水记录的超市数据库中，顾客单买了面包的有 40 条，单买了黄油的有 50 条，而面包和黄油被同时购买了 30 条。<br>
请你手算出提取的关联规则 <code>{面包} -> {黄油}</code> 的 Support, Confidence, 以及证明这是否是一个具有正向促进意义的强相关推荐规则 (Lift)。</p>
<p><strong>【解答】</strong>:<br>
- <strong>Support</strong> = 包含该组合的总条数 / 总交易数 = $\\frac{30}{100} = 30\\%$<br>
- <strong>Confidence</strong> = $Support(面包 \cup 黄油) / Support(面包)$ = $\\frac{30}{40} = 75\\%$。这意味着买面包的人里高达 75% 会买黄油。<br>
- 这个推荐真的有用吗？必须算 <strong>Lift (提升度)</strong>：$\\frac{P(面包 \cup 黄油)}{P(面包) \\times P(黄油)} = \\frac{0.3}{0.4 \\times 0.5} = \\frac{0.3}{0.2} = 1.5$。<br>
- 结论：因为 <strong>Lift = 1.5 > 1</strong>，这证明买面包确实正面、强烈地促进了黄油的销售，绝非巧合现象。这是一个极其有用的强黄金规则！</p>
<p><strong>【Q2. 算法填空题】</strong>在 Apriori 算法从 2-项频繁集生成 3-项候选集的极度凶残剪枝 (Prune Step) 过程中，假设机器刚刚拼凑出了一个崭新的候选集 $C_3 = \\{苹果, 橙子, 香蕉\\}$，在去数据库真正拉取全表点数前，系统会立刻回过头去检查它所有的 2维非空子集：$\\_{______}\\_$，$\\_{______}\\_$，$\\_{______}\\_$。只要发现其中有任何一个连上次的 $L_2$ 频次底线都没过，系统就会立马行刑砍掉这个 $C_3$。这是因为该算法最核心的 $\\_{______}\\_$ 性质。</p>
<p><strong>【解答】</strong>: 答案填入依次为 <code>{苹果, 橙子}</code>、<code>{苹果, 香蕉}</code>、<code>{橙子, 香蕉}</code>。因为该算法最核心的 <code>Anti-monotone (反单调性 / 向上封闭性)</code> 性质判定：一个组合要是想火爆，拆成任何小分队都必须火爆。</p>
"""

m09 = """
<h1>MODULE 09: Partitioning & Hierarchical Methods (划分与层次聚类)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>ClusterAnalysis_Part.pptx</code> & <code>ClusterAnalysis_Hier.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>聚类思想的根本区别、K-Means质心更新推导及其面对非凸形状与异常点的脆弱性、三种层次聚合的 Linkage 计算差异与不可逆缺陷、WEKA 中的具体实现探讨。</div>
<h2>1. 聚类分析背景与划分流派 (Partitioning Methods)</h2>
<p>分类算法拥有预先标记的 Ground Truth，而聚类 (Clustering) 是纯粹的无监督学习，它盲人摸象式地试图根据对象自身的内部属性（如几何距离），将数据集分割为 $K$ 个组，使得 <strong>簇内高度内聚，簇间极其疏远</strong>。</p>

<h2>2. WEKA 中的 SimpleKMeans 质心吸扯流引擎</h2>
<p>作为最经典的中心辐射流算法，它在 WEKA 中的名为 <strong>SimpleKMeans</strong>，其每轮迭代步骤被固化为如下逻辑：</p>
<ol>
    <li>在茫茫高维数据海中，闭着眼睛 <strong>随机空投挑出</strong> $K$ 个倒霉的特征点作为初始的镇山之石 (Centroids)。</li>
    <li><strong>[Assignment 无脑收编环节]</strong>：它跑遍全服所有的点，逼着所有人用欧氏距离标尺去量自己跟这 $K$ 个大哥的物理距离，谁离得最近，它立刻就被强行盖下烙印，归入对应质心的旗下领地 $C_i$。</li>
    <li><strong>[Update 权力移交洗牌]</strong>：全员刚刚认完阵营，K-Means 立刻在各个局部阵营里内部收割统计。它算出每一个阵营大圆内所有小兵小将特征坐标位置的 <strong>绝对中心原点 (几何平均坐标 Mean, $\\mu_i$)</strong>。接着，原本的大哥直接被废黜卸任，质心信物被移交给这个全新的虚无中心均值点！</li>
    <li>由于新大哥的位置发生了平移跑动，原先边防线上的人一照镜子，发现隔壁家的那个新大哥竟然离自己更近！这引发了疯狂的阵营哗变倒戈。</li>
    <li>这套 Assignment 和 Update 双重循环永动不休。直到某一次 Update 后，新均值跟老质心竟然丝毫不差一动没动，整个阵营再无一人叛变（即整体离差平方和 <strong>$SSE$ 方差极小值彻底收敛固化 Converged</strong>），系统强制鸣金停跑，吐出最终划分格局。</li>
</ol>

<h3>SimpleKMeans 的四大致命死穴 (Limitations) 考点：</h3>
<ul>
    <li><strong>对初始落点极度敏感</strong>：纯随机初始化很容易让两个质心降落在同一个自然簇中，导致它陷入毫无意义的 <strong>局部最优解 (Local Optimum)</strong>。在 WEKA 中，SimpleKMeans 往往需要配合 K-Means++ 的聪明初始化或进行多次随机重启取最优。</li>
    <li><strong>无法捕捉非球形结构 (Non-convex bias)</strong>：K-Means 的距离分配和均值运算，使其潜意识里认定簇的形状必须是完美的球体/凸形。面对现实世界中环套环、S形、或两条平行线形态的簇，它会直接一刀切碎，彻底崩坏。</li>
    <li><strong>大小悬殊与同心质心崩溃</strong>：如果数据中一个小簇被包裹在一个大簇的同一个引力中心周围（如同心圆），或者两簇大小相差巨大，它会强行切分导致失真。</li>
    <li><strong>对离群噪声 (Outliers) 毫无抵抗力</strong>：一个处于几千公里之外的错误噪点被强行归入某簇后，在计算均值时会像杠杆一样极其剧烈地将整个质心拉扯向深渊。<br>
    <strong>应对方案：K-Medoids 算法 (PAM)</strong>，它不允许计算虚空的均值质心，而是强制挑选当前簇内最居中的那个 <strong>真实的实体样本</strong> 作为中心。这就好比中位数对抗平均数，极大压制了离群噪点的杠杆效应。</li>
</ul>

<h2>3. 层次流派的雪人堆叠法 (Hierarchical Clustering)</h2>
<p>划分聚类生硬地将数据拍成扁平的 K 块，而层次聚类无需你提前告诉它 K 是多少。在 WEKA 里被归为 <code>HierarchicalClusterer</code>。它自底向上 <strong>凝聚 (Agglomerative, AGNES)</strong>，一开始每个人都是独狼。每次循环，它都在天地间寻找看起来距离最近的两个组，将它们吞并结合。一步步直到最终所有的数据汇流成一棵宏大的 <strong>层次进化树 (Dendrogram)</strong>。随后人只要随便在高度上砍一刀，想切几个簇就切出几个簇。</p>
<h3>核心考点：决定两个多边形簇群之间“距离”的三大几何判定法：</h3>
<ul>
    <li><strong>Single-linkage (单链法 / 最短距离)</strong>：取簇 A 与簇 B 中互相靠得 <strong>最近</strong> 的那对成员作为代表，以此计算两个庞大簇群的间距：$d(A, B) = \\min_{x \\in A, y \\in B} d(x, y)$。<br>
    <strong>巨大缺陷</strong>：这会使得算法极度偏爱长条形、蛇形蔓延的聚类。如果两团互不相干的人群中间碰巧散落了几个噪点，单链法会通过这些噪点“搭桥”，将两组毫不相干的人硬生生粘成一团。这就是著名的 <strong>链式效应 (Chaining phenomenon)</strong>。从图论上讲，这种聚合过程完全等价于 Kruskal 的 <strong>最小生成树 (MST)</strong> 连线。</li>
    <li><strong>Complete-linkage (全链法 / 最长距离)</strong>：极其严格，取两个簇中互相离得 <strong>最远、最具敌意</strong> 的那对成员作为代表来计算距离：$d(A, B) = \\max_{x \\in A, y \\in B} d(x, y)$。这强迫产出的簇必须是极其紧凑的球形团块，但也极易受到游离在簇边缘的极端噪点的影响。计算开销也变得极其昂贵。</li>
    <li><strong>Ward's Method (沃德方法)</strong>：它完全抛弃了点对点的死板距离。它假想：如果我把这两个簇合并，整体系统的变异度（方差增量）会增加多少？它总是优先合并那些合并后使得总系统混乱度增加最小的、看起来最门当户对的两个组。</li>
</ul>
<p><strong>层次聚类无法挽回的悲剧</strong>：无论你用上述哪种链式方法，层次聚类具有一种宿命论缺陷——<strong>不可逆转 (Irreversible)</strong>。一旦算法在早期的底层迭代中做出了错误的合并决定，它将永远无法拆开这个错误组合，错上加错直至顶端。</p>

<h3>🎯 Mock Exam 经典例题</h3>
<p><strong>【Q1. 算法流演练题】</strong>简述 K-Means 算法每一轮核心迭代中所执行的两个具体数学推导步骤，以及它是如何判断应该“收工停跑”的？</p>
<p><strong>【解答】</strong>:<br>
- 步骤一 <strong>[Assignment]</strong>：遍历空间中所有数据点 $x$，分别计算它们与当前所有 $K$ 个质心坐标 $\\mu$ 之间的欧氏距离。寻找最短距离，强行将其分派到对应质心的领地下，即 $C_i^{(t)} = \\{x : ||x - \\mu_i^{(t)}|| \\le ||x - \\mu_j^{(t)}||\\}$。<br>
- 步骤二 <strong>[Update]</strong>：待全员分配完毕，重新收割各个领地 $C_i$ 内所有的点群，求出它们的矢量坐标几何平均值 $\\mu_i^{(t+1)} = \\frac{1}{|C_i^{(t)}|} \\sum_{x \\in C_i^{(t)}} x$，以此新坐标作为下一轮质心。<br>
- <strong>如何停止</strong>：由于其本质目标是不断压缩局部方差和指标 (SSE)，一旦在最新的迭代中，所有的质心坐标都不再发生一毫米的挪动偏移（或者说没有哪怕一个人发生阵营跳槽叛变），代表着 SSE 已经彻底塌陷到当前可能的谷底（收敛 Converged），算法即刻抛出终盘停工。</p>
<p><strong>【Q2. 概念辨析】</strong>对于一团完全呈现出如同心圆一般的非凸形空间数据集（外围是一圈庞大的环形点阵，核心是一个密集球体），如果使用 Single-linkage 层次聚类能够将它们正常剥离吗？如果换用 K-Means 呢？</p>
<p><strong>【解答】</strong>:<br>
- <strong>Single-linkage</strong>：只要外围的环内部紧紧连绵相接，且环与核心球体之间保持了极其微弱的宽阔断层间距，单链法就能通过最近点搭桥，完美像拨桔子皮一样把它们分离出内外两圈！<br>
- <strong>K-Means</strong>：彻底被耍得团团转崩溃。因为它信奉中心放射引力法则，既然两团分布共享着一模一样的空间中心位置，它的质心会被这股对流吸扯在一起，最终强行挥舞直板大刀将同心圆像切西瓜一样切成两半边，闹出最大的笑话。</p>
"""

m10 = """
<h1>MODULE 10: Density-Based Clustering (基于密度的聚类)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>ClusterAnalysis_DB.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>DBSCAN 算法中三大实体（核心、边界、噪声）的严格定义、密度直达与相连的连通性、基于密度如何完美规避 K-Means 的形状限制、以及为了应对变密度问题而进化出的 OPTICS 可达性图技术。</div>
<h2>1. 打破球形迷信：密度聚类的哲学动机</h2>
<p>K-Means 只认同以几何质心为圆心的完美球状簇，层次 Single-linkage 则容易被随机散布的孤立噪点骗去搭桥。有没有一种方法，既能识别像盘山公路一样弯曲的任意非凸形状簇 (Non-convex)，又能在面对漫天噪点时铁面无私地将它们无视？<br>
答案是 <strong>Density-Based (基于密度的聚类)</strong>。该流派认为，簇的本质是空间中被一大片“荒漠区域”隔开的一块块“高密度人口聚集区”。只要人口密度一直连绵不断，无论这片居住区形状多么奇特怪异，它都算作同一个簇群。</p>

<h2>2. DBSCAN 雷达波探测大阵执行流</h2>
<p>在 WEKA 中的 <strong>DBSCAN</strong> 高度依赖两个至关重要的雷达控制台参数：<br>
1. <strong>Eps ($\\epsilon$)</strong>：探测雷达探测距离的极限辐射半径。<br>
2. <strong>MinPts</strong>：想要拥有资格成立一个新山头“据点”，在你自己这片雷达范围内必须凑齐的最低人口底线要求（包含你自己）。</p>
<h3>数据实体的三六九等阶级定性法则：</h3>
<ul>
    <li><strong>Core points (核心点)</strong>：最强王者级。如果某个样本点拉下开关让 Eps 雷达旋转一圈扫描，统计发现被照到的圈内人数 $\\ge MinPts$，恭喜它，立刻黄袍加身晋升为“核心点”，成为搭建市中心商圈的绝对基石。</li>
    <li><strong>Border points (边界点)</strong>：随风倒边缘配角。它拉开雷达发现圈内人数惨淡，根本不够 MinPts 指标，完全没有自立门户的资质。但是，极其幸运的是，它 <strong>刚好被包裹在了某个王者级核心点打出来的强力 Eps 辐射圈内</strong>！因此，它被核心点直接顺手牵羊强行收编，委派去看守该聚类簇的最外围边界护城河，并被尊称为“边界点”。</li>
    <li><strong>Noise (噪声点)</strong>：惨遭流放者。自己雷达圈内人数极少废墟一片，同时也悲惨地没有被大地图上任何一个核心点的势力范围所搭救覆盖。DBSCAN 直接将其判定打上冷酷的 Noise 废品标签彻底封存。这就是它 <strong>天生拥有强大抗噪鲁棒性、不被单点污染带偏整体</strong> 的核心奥秘所在。</li>
</ul>
<h3>组建庞大帝国的顺藤摸瓜流（DBSCAN完整执行环）：</h3>
<ol>
    <li>系统启动！一上来二话不说，将地图上所有几百万个点全服打上 <strong>Unvisited (未探索蒙昧地带)</strong> 的冰冷标记。然后，神明降临，随便凭运气闭眼随机挑中并唤醒点 A，把它戳上 Visited 面纱。</li>
    <li>点 A 领命立刻按下探照灯，向着四面八方扫出一波距离 $\\le Eps$ 的邻域雷达图。</li>
    <li>系统冰冷裁决：如果点 A 雷达图里的猎物总数不幸落后于 $MinPts$ 指标，系统直接当头盖下 <strong>Noise (噪声垃圾)</strong> 的印章扔一边（注意这只是暂时的冷宫，说不定后面它走狗屎运会被其他巨佬核心扫中变成边界护城河被捞出来）。</li>
    <li>但如果大喜！点 A 扫出的人马浩荡庞大 <strong>$\\ge MinPts$</strong>，那这股燎原之火就正式烧起来了。点 A 被册封 <strong>Core Point (神圣核心点)</strong>。系统为了它专门隆重奠基拔地而起一座崭新的大本营营寨——命名为全新的大簇 $C$！然后理所应当地，点 A 和它刚才扫到的所有眼皮子底下的直接邻居小弟全被扯着塞进了营寨 $C$。</li>
    <li>可怕的病毒级传销扩张 <strong>Density-reachable (密度可达传播网)</strong> 开启！<br>刚刚被点 A 拉进伙的邻居如果依然是 Unvisited，也被强制开启探照灯继续找新邻居。只要这些个下家邻居自己争气，探照出来的人头居然也能突破 $MinPts$，那么下家也能被追封为次级核心！下家新招揽来的一串人马直接跟着大哥一起连滚带爬地统统拉入到当初的大本营 $C$ 里去！<br>
    这一套像击鼓传花一样不断通过 <strong>密度相连 (Density-connectedness)</strong> 传导出去的网络，最后会将整个地图上连绵不绝互相勾搭拉拢的成千上万个核心基站全拼在一起，再加上那些没出息只够挂着当边境墙的边界点小弟，就化合成了一个犹如毒液般千姿百态、外人根本无法用圆规画出来的巨大畸形独家异种簇群！</li>
    <li>当这把连绵大火烧断了气彻底扑腾不出去后，系统才换一口气，回头去剩下全服还在沉睡的无尽 Unvisited 人群里重现挑一个幸运儿唤醒当下一个点，机械循环回步骤1。直到天下地图上不再有 Unvisited 者，系统光辉下发“Game Over”。</li>
</ol>

<h2>3. 变密度的痛点与 OPTICS 的降维打击</h2>
<p><strong>DBSCAN 的死穴</strong>：它对 Eps 和 MinPts 两个参数的依赖达到了病态的程度。<br>
想象一个场景：城市中心的高端住宅区密集无比，而郊区的平房区相对稀疏。如果设定极小的 Eps，城市中心完美分群，但郊区的居民将全被判定为孤立噪声点；如果设定极大的 Eps，郊区居民成功建簇，但整个城市中心的多个截然不同的小区会被极度宽泛的阈值直接溶解成一个大黑洞。<br>
<strong>结论：DBSCAN 在应对不同密度分布混合的数据集时，单一全局阈值会导致表现极度拉垮。</strong></p>
<h3>OPTICS (Ordering Points To Identify the Clustering Structure)</h3>
<p>为了拯救变密度问题，OPTICS 进行了降维打击。它不再强求去画出一个个明确的聚类边界圈，而是把多维数据全部排序成一个一维序列，并为每个点计算出两个动态距离：<br>
1. <strong>Core-distance (核心距离)</strong>：能让它刚满 MinPts 及格线的最短最小半径：$Core\\_distance(p) = \\min_{N_{Eps}(p) \\ge MinPts} \\text{distance to } MinPts\\text{-th nearest neighbor}$。<br>
2. <strong>Reachability-distance (可达距离)</strong>：别的点想要够到它需要跨越的最少距离（不能小于它自己的核心距离）：$Reachability\\_distance(p, o) = \\max(Core\\_distance(o), d(p, o))$。</p>
<p>基于这个序列，OPTICS 绘制出著名的 <strong>Reachability Plot (可达性图)</strong>。<br>
在这个二维柱状图中，所有点连绵起伏。那些数值很低、深陷下去的 <strong>“山谷 (Valleys)”</strong>，就代表了距离极短、相互拥挤的密集人口簇；而那些突兀的高峰，就是稀疏区域或噪点。<br>
<strong>最强大的地方在于</strong>：即便是城市中心的超深谷（超密集簇）和郊区的浅谷（一般密集簇），人类或后续切分算法可以直接根据图表上的波谷轮廓将它们分别切分出来，完全不再需要去纠结那个进退两难的全局固定边界 Eps！</p>

<h3>🎯 Mock Exam 经典例题</h3>
<p><strong>【Q1. 判定填空题】</strong>在执行 DBSCAN 聚类中，设定 $Eps=1.5$, $MinPts=4$。此时侦测到点 A 的 1.5 邻域内恰好只装进了 2 个点（自己和点 B）。同时，点 B 是一个名副其实的超级地王（其圈内有 10 个人），那么请问点 A 此时该挂上 <strong>____</strong> 的牌子，而点 B 将被尊称为 <strong>____</strong>？</p>
<p><strong>【解答】</strong>: 填入 <code>Border point (边界点)</code> 和 <code>Core point (核心点)</code>。<br>
- 点 B 有 10 个人（远超门槛的 4 个人），所以 B 是毫无悬念的核心大员。<br>
- 点 A 自己周围门庭冷落（只有 2 个），根本不及格 $MinPts=4$ 的要求，无法当上核心。但是它极其好命地撞进了点 B 这位大员的 1.5 半径庇护伞内，它马上就可以被纳入到大员构建起来的浩瀚簇群大军当中，安心当个外围城墙——Border point。</p>
<p><strong>【Q2. 解析原理题】</strong>如果我把 DBSCAN 的 $MinPts$ 极端地设置成 1，请问此时的 DBSCAN 实际上退化成了我们在另一章中学到的哪一种经典距离聚类方法？请解释一下原理。</p>
<p><strong>【解答】</strong>: 彻底退化为了 <strong>Single-linkage Agglomerative (单链法层次聚类)</strong>！<br>
因为只要 $MinPts=1$，那么在整个宇宙空间内，只要存在点，任何单个孤零零的点都能毫不费力地宣称自己是一个 Core point。这就使得 DBSCAN 中所有复杂严苛的护城河机制全部瘫痪失效。接下来的情况将直接变成：只要任何两个人之间的距离在 $Eps$ 以内，它们就因为同属于核心点而顺理成章地勾搭在了一起。无论形状怎么蜿蜒，最后全连通成了一个如同 MST (最小生成树) 牵线的巨大毛线团。这完全就是单链法的灾难化翻版演绎。</p>
"""

m11 = """
<h1>MODULE 11: Cluster Analysis Evaluation (聚类评估与指控)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>ClusterAnalysis_EV.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>聚类分析有效性前提、Elbow肘部法则的寻找逻辑、Silhouette轮廓系数中内聚和分离的抗争、以及外在指标 B-Cubed 评估与 WEKA 的错误率折算。</div>
<h2>1. 聚类评估的三重天审视</h2>
<p>与监督学习可以通过交叉验证算出确凿无疑的准确率不同，评估在黑暗中摸索的聚类是一门充满主观性与争议的玄学。我们必须从三个层次进行拷问：</p>
<ul>
    <li><strong>Tendency (聚类趋势判定)</strong>：最先决的条件。拿来的这批数据，它们真的在底层逻辑中存在所谓的“成簇聚集”结构吗？有没有可能它们就像布朗运动一样完全均匀、毫无统计显著性地随机散布在多维空间中？如果不经过霍普金斯统计量等方法检验，强行对一堆随机噪点去聚类，纯粹是自欺欺人。</li>
    <li><strong>Quantity (簇量探查)</strong>：比如最令人头疼的 K-Means 中的那个 $K$ 值，究竟存在几个天然簇群？</li>
    <li><strong>Quality (聚类品质)</strong>：最终划分出的群体，内部的人是不是脾气相投（内聚），群体与群体之间是不是老死不相往来（分离）？</li>
</ul>
<h2>2. Intrinsic measures (内在指标：在没有真实答案情况下的内斗)</h2>
<p>当现实业务（如给商场几百万客户做分群推荐）不存在标准的 Ground Truth 标签时，算法的好坏只能依赖空间几何形态来自我验证。</p>
<ul>
    <li><strong>Elbow Method (肘部法则) 探查 K 值</strong>：<br>
    它主要配合 K-Means 这种致力于最小化误差平方和 (Sum of Squared Errors, SSE) 的算法。你强行把 K 从 1 开始往上推，分别计算总的 SSE。随着簇越来越多，大家越分越细，SSE 必定机械式地越来越小。<br>
    当从 $K=2$ 到 $K=3$ 时，如果是切中要害，SSE 会出现瀑布式断崖下降；当超越真实的簇数量继续往上切时，下降的斜率就会像被强行刹车一样变得极为平缓。在曲线上寻找这个明显发生转折的 <strong>“肘部 (Elbow / 拐点)”</strong>，这就是数据自身在呐喊的最佳 $K$ 值。<br>
    <strong>致命弱点</strong>：很多现实数据的分布黏糊不清，曲线极为平滑，根本找不到所谓的完美肘部。</li>
    
    <li><strong>Silhouette Coefficient (轮廓系数) 评价质量</strong>：<br>
    它是检验单个样本点内心挣扎的最精准度量。假设我们考察某个倒霉样本 $i$：<br>
    第一步：计算 $i$ 到它 <strong>自家簇群</strong> 中所有兄弟的平均距离，这叫 $a(i)$，它衡量了内聚度。显然 $a(i)$ 越小，说明它跟自家兄弟越亲密无间。<br>
    第二步：计算 $i$ 到所有 <strong>隔壁别家簇群</strong> 中的人的平均距离，取其中平均距离最小的那个簇群作为它最可能的“小三备胎簇”，这个距离叫做 $b(i)$，它衡量了分离度。显然 $b(i)$ 越大，说明它离别家的人越遥不可及，界限越分明。<br>
    公式为：$s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}$。<br>
    <strong>结果判读（必考）</strong>：轮廓系数的游走范围严防死守在 <strong>[-1, 1]</strong>。<br>
    当结果 <strong>极其接近 1</strong>，代表 $b \\gg a$，即离别的簇极远，跟自家兄弟抱得极紧，这是极其完美的聚类状态。<br>
    当结果为 <strong>0</strong> 时，说明 $a = b$，该样本尴尬地骑在两个簇群边缘的边界线上，不知所措。<br>
    当结果 <strong>滑入负数地狱 (接近 -1)</strong>，说明 $a \\gg b$，它离自家人的距离甚至比离别人家还远！这意味着聚类算法犯了不可饶恕的分配错误，将这个样本强行拖错了阵营。</li>
</ul>
<h2>3. Extrinsic measures (外在指标：用上帝视角的标签审判)</h2>
<p>在算法考试竞赛或学术验证中，我们往往拥有隐瞒给算法的真实分类标签 (Ground Truth Labels)，此时可以直接用它们充当法官进行终审裁决。</p>
<ul>
    <li><strong>B-Cubed precision and recall (基于元素的评价)</strong>：<br>
    不再从宏观混淆矩阵去看，而是微观到每两个个体的绑定关系。对于任何一个具体的样本，观察它所在的聚类小圈子，看看圈内有多少人 <strong>真实类别也确实与它一致</strong>，这就是它在这个簇里的 Precision；再看看总体中跟它同样真实类别的人，有多少被 <strong>成功招揽进了这个同样的簇里</strong>，这是 Recall。将全员加权平均得出。</li>
    <li><strong>Classes to clusters evaluation (WEKA 映射对位黑魔法执行流)</strong>：<br>
    面对无监督算法吐出的一堆没有任何含义的数字名字如 Cluster 0, Cluster 1, Cluster 2 蒙面群落，你由于不知道它们究竟对应原先实际的猫、狗还是猪的类标，你在 WEKA 里强制运行这套魔法。<br>
    WEKA 此时会在底层引擎中暗中强行驱动 <strong>贪婪排列组合大阵 (Hungarian or Greedy Matching)</strong>，硬是在一堆杂乱盲开出来的聚类编号盘子里，千方百计地拨弄配对！直到它终于找准了一条唯一能让真实大类样本和机器瞎编出来的群落标签对齐人数最多、重叠度最庞大的完美金线。<br>
    一旦这层翻译加密映射被破解，一切豁然开朗。剩下那些不管怎么画圈都塞不进去映射格子的离群异端们，自然就被系统等效转化为了传统监督战场的那些打靶失败的 <strong>错分炮灰 (Classification Error Rate)</strong>，从而在一套原本不讲武德的聚类引擎里强硬计算出了惊世骇俗的最直观精度报错比例。</li>
</ul>
<h3>🎯 Mock Exam 经典例题</h3>
<p><strong>【Q1. 极限推理与计算题】</strong>在一个聚类任务中，提取出了某个特定数据点 $x$ 的距离数据：它到本簇兄弟同伴们的平均距离是 5，它到距离它最近的那个隔壁备胎大簇里的敌人们的平均距离竟然只有区区 1 ！请套用 Silhouette Coefficient 轮廓系数公式计算此样本点的悲惨指数，并判定该点是不是被算法彻底分错坑位了。</p>
<p><strong>【解答】</strong>:<br>
- 提取已知量：内聚极距 $a=5$，备胎分离极距 $b=1$。<br>
- 代入无敌审判公式：$s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))} = \\frac{1 - 5}{\\max(5, 1)} = \\frac{-4}{5} = -0.8$<br>
- <strong>终极判罚</strong>：由于轮廓系数达到了负向的极高水准 $-0.8$（已经逼近理论的深渊底线 $-1$），它证明这颗倒霉点跟自己“阵营”里的人隔了足足五条街，却跟隔壁敌营的人勾肩搭背挤在一起（距离仅有1），这意味着本次聚类算法在这里犯下了严重极其滑稽的低级归属划分错误。</p>
<p><strong>【Q2. WEKA 判定原理】</strong>当手头有非常严谨的 <code>Real_Labels</code> 时，你在 WEKA 里强制运行了 K-Means 这种不受人类控制乱编簇号的盲聚算法。跑完之后，WEKA 显示 <code>Incorrectly clustered instances : 15.0%</code>。请问 WEKA 是用了何种鬼斧神工般的外挂策略，能把完全没有对齐名字的 Cluster0, Cluster1, ... 硬生生给强制转化为类同监督分类问题的精准度 Error rate？</p>
<p><strong>【解答】</strong>: WEKA 使用了著名的 <strong>Classes to clusters evaluation 匹配大阵</strong>。<br>
由于盲跑出来的簇叫什么根本无关紧要（它可能把猫叫 Cluster 0，狗叫 Cluster 1），WEKA 在后天直接强行穷举了所有的重组合规序列，寻找到了那条唯一能让真实大类样本和机器瞎编出来的群落对齐人数最多、重叠度最高的完美映射线。<br>
既然对齐了，剩下的没有挤进对齐光圈里的那些离群异散者，自然就等效转化为了分类战场上的那些错分炮灰，从而精妙地直接得出了这极其直观的 15.0% 等效错误率。</p>
"""

m12 = """
<h1>MODULE 12: Data Warehouse & OLAP (数据仓库的维度之战)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>DataWarehouse_DCOLAP.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>OLTP与OLAP两种极端数据库的理念割裂、基于 Dimension 与 Fact 的星型架构维度建模、Lattice 晶格组合爆炸的挑战、以及掌控切分数据的四大核心 OLAP 降维/升维手术刀操作。</div>
<h2>1. 业务型 OLTP 遭遇的雪崩与 OLAP 的突围</h2>
<p>传统的 <strong>OLTP (Online Transaction Processing, 联机事务处理)</strong> 数据库（例如你购买奶茶支付时极速响应的底层 MySQL 关系型库），被设计成了面向高度细分的单条交易流水的极端插入、删除和更新 (CRUD)。但当企业 CEO 要求查询“调取过去三年，整个华南大区每月各类饮品的销售额环比增长图表”时，由于关系型数据库没有任何跨表长线预聚合，这种长达千万行级别的全局 <code>GROUP BY</code> 联合查询操作将像绞肉机一样瞬间榨干底层 CPU 算力，直接导致甚至几小时的长时锁表，拖垮整个公司的核心结算系统。<br>
为了解救业务系统的生死存亡，脱胎换骨的 <strong>Data Warehouse (数据仓库)</strong> 应运而生。它引入了 <strong>OLAP (Online Analytical Processing, 联机分析处理)</strong>。其最高哲学是不再响应细碎的单次修改，而是通过抽取离线历史数据，<strong>提前按照成千上万个分析组合把统计口径的最终总和全部暴力计算、存储结块在多维的 Data Cube (数据立方体) 中</strong>，使得老板在查几十年总账时，其本质不过是调取了一个已被预先算死存放在某个维度交点坐标上的数字，实现毫秒级查询神话。</p>
<h2>2. 维度建模 (Dimension Modeling) 与 Lattice 晶格法则</h2>
<p>在数据仓库的星型模型 (Star Schema) 中，复杂的关系表被极其残酷地砍为了两级对立的抽象阵营：</p>
<ul>
    <li><strong>Dimension (维度框架)</strong>：构成观察世界的 $N$ 个基准坐标轴体系。如：时间维度 (Time)、地点维度 (Location)、货品维度 (Item)。这是用来 <code>GROUP BY</code> 和条件检索的刻度尺。</li>
    <li><strong>Fact / Measure (事实度量)</strong>：真正被记录在坐标系正中央，随着不同条件切割被加加减减的核心数值目标。如：总售出数量 (Units sold)、带来营收 (Dollars)。</li>
</ul>
<p><strong>Lattice Structure (晶格重力网络)</strong>：<br>
千万不要以为 Data Cube 就是简简单单的一个三维魔方块！如果你的维度有 $N$ 个，它其实包含了整整 $2^N$ 个不同聚合组合维度的 <strong>Cuboids (方体切面)</strong> 集合。从只看“时间”这1个维度（1-D cuboid）、到把“时间+地点”组合起来看（2-D cuboid）、甚至一直到 0-D 的 <strong>Apex Cuboid</strong>（不加任何前置查询条件，就是算一个全地球所有年份全品类物品混在一起加总的终极数字），最终形成了一张极其密集复杂的上下连通的 <strong>Lattice (晶格网络)</strong>。</p>
<h2>3. 凌驾维度之上：四大 OLAP 神之手操作</h2>
<p>如何在这个复杂的魔方矩阵里自由穿梭？借助预先定义的 Concept Hierarchies (概念上下级层级：比如 街道 $\\rightarrow$ 城市 $\\rightarrow$ 省份 $\\rightarrow$ 国家)，分析师可以通过四个动作随意拆解维度：</p>
<ul>
    <li><strong>Roll-up (上卷，或称泛化聚合)</strong>：<br>
        动作：消灭某个维度，或者顺着概念层级 <strong>向上攀升归并</strong>。<br>
        效果：原本在看每个街道具体每日卖了多少水，一记 Roll-up 后，把所有的天合成了月，把街道合成了整个广东省，数据变得高度宏观，维度缩减，细节消失。</li>
    <li><strong>Drill-down (下钻，或称细化拆解)</strong>：<br>
        动作：Roll-up 的绝对反向剥离。强行拽入全新的横切维度，或者沿着概念层级 <strong>向下沉入微观</strong>。<br>
        效果：CEO 看着广东省三月份的整体惨淡业绩拍桌子，通过一记 Drill-down，直接把这个数字瞬间撕裂回了几十个市级维度，再配上周级别的颗粒度，找出具体到底是哪个底层环节在拖后腿，数据细节急剧膨胀爆出。</li>
    <li><strong>Slice (横向一刀切片)</strong>：<br>
        动作：在漫长的任意 <strong>单独 1 个维度上</strong>，冷酷地固定锁死一个特定值进行平行切割。<br>
        效果：你面前的 3D 魔方被一刀劈下了一张平坦的 2D 截面。比如，万般维度里，强制切下 <code>"Time = 2025年2月"</code>，从此所有展露在你面前的数据报表全部被死死封印在这个单一月份中流转。</li>
    <li><strong>Dice (切块提取核心阵地)</strong>：<br>
        动作：更为高阶精细的打击，在 <strong>多个交叉维度上同时划定范围区间</strong>。<br>
        效果：从巨大无比的魔方中，抠出了内部一块高度收缩聚焦的子立方体阵列（比如，只圈出 <code>"Time = 第一季度"</code> AND <code>"Location = 北上广"</code> AND <code>"Item = 电子产品"</code> 这批组合阵列进行观摩）。</li>
</ul>
<h3>🎯 Mock Exam 经典例题</h3>
<p><strong>【Q1. 动作辨析匹配题】</strong>针对在三维维度中（Time, Location, Item）记录着海量流水事实记录的系统。请判断以下两波老板发出的变态指令，分别动用了四大天王（Roll-up, Drill-down, Slice, Dice）中的哪一招绝学：</p>
<p>A老板咆哮：“我不想看你们这些细碎的各个月份数字，也别给我分产品和手机型号，全部给我合并汇总成全年的整体营收额！”<br>
B老板要求：“给我只单独把广州市和深州市第一季度的手机品类销量数据给我抠出来做成报表，其他我不想看！”</p>
<p><strong>【解答】</strong>:<br>
- <strong>A 老板 -> Roll-up (上卷归约)</strong>。他强硬地抛弃了繁杂的商品细分维度，同时对日期大刀阔斧地顺着时间层级拔高合并，所有的数据变得庞大而概括化。<br>
- <strong>B 老板 -> Dice (强力切块)</strong>。他不仅缩窄限定了特定的地点（广/深两市双向切割），又把时间死死锁在了一季度内，同时还单独只提取手机这个具体品类。这种在多重维度上划定子区间的精密组合手术刀操作，就是名副其实的终极切块 (Dice)。</p>
"""

m13 = """
<h1>MODULE 13: Data Cube Computation (立方体运算的 BUC 裁决与闭环压缩)</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>DataCubeComputation_BUC.pptx</code> & <code>DataCubeComputation_CCIC.pptx</code></p>
<div class="highlight"><strong>核心考点：</strong>如何辨别聚集函数不可调和的计算性质、维数组合爆炸导致 Iceberg Cube 的诞生，以及 BUC 算法以自顶向下 (Apex起步) 为名，强行利用“下钻分化数据递减的单调性”完成整枝无损大裁剪的极高频算法灵魂。</div>
<h2>1. 聚集函数 (Aggregate Functions) 的计算阶级鸿沟</h2>
<p>在利用魔方的上层结果向下快速合并推导父辈汇总之时，底层聚合函数遵循着严苛的层级定律：</p>
<ul>
    <li><strong>Distributive (纯正分配型：极其完美)</strong>：<br>
    涵盖 <code>Count</code>, <code>Sum</code>, <code>Min</code>, <code>Max</code>。它们的恩赐在于，如果老板要查整个广东省的总销售额（省级父节点），不需要再去逐条读取原始流水线，只要直接把深圳、广州、珠海等几个子节点的 <code>Sum</code> 再次做一次 <code>Sum</code> 操作，结果丝毫不损，极其快哉。</li>
    <li><strong>Algebraic (代数寄生型：勉强能算)</strong>：<br>
    典型如 <code>Average</code>。你绝对不能拿深圳均值加上广州均值除以2当做广东省的均值，这在数学上是极其荒谬的。但万幸的是，如果你在底层保存了两个不起眼的 Distributive 辅助标量——各自区域的 <code>Sum</code> 和 <code>Count</code>——那你就可以像施展魔法一样通过 $\\frac{\\sum Sum_i}{\\sum Count_i}$ 瞬间完成父辈代数还原。</li>
    <li><strong>Holistic (整体地狱型：算力黑洞)</strong>：<br>
    最臭名昭著的是 <code>Median (中位数)</code>、<code>Mode (众数)</code>、以及各种极度复杂的 <code>Rank</code>。如果你算出了北京和上海各自人群财富的中位数，抱歉，对于求全国财富的中位数，它们没有半毛钱帮助！你唯一的出路只能是极其悲惨地把北京上海所有千万人头的底层原始数据再次全盘拉取出来，混在一起重新全部大排序再抽底。这对海量层级汇总的数据仓库而言是算力毁灭级别。</li>
</ul>
<h2>2. 维数诅咒的绝望与 Iceberg Cube 的救赎法则</h2>
<p>在面对成百上千的高维稀疏表格时，组合的 <em>Full Data Cube</em> 节点将以 $2^N$ 的指数海啸级膨胀爆炸。<br>
但幸运的是，在大数据的商业现实中，那些切分得过细、总计数 $Count \\le 1$ 的零星随机偶然搭配网格点，根本不具备挖掘长线统计规律的价值。<br>
<strong>Iceberg Cube (冰山立方体)</strong> 因此横空出世：强制引入一层硬性闸门——<strong>Iceberg condition (冰山阈值限制，通常为 Min_Support / 最小出现频次数)</strong>。如同浩瀚的数据海洋，系统只被允许去计算、汇总和留存那些高频发生、足以浮出水面之上的极少部分高价值网格阵列集，其余海平面下的稀碎偶然杂音直接被抛弃不计算，从而极其恐怖地削减计算时空开销。</p>

<h2>3. 冰山上的裁决利刃：BUC 数据立方体心决执行流</h2>
<p>这是一个名字具有极大欺骗性的神级操作体系。虽然名冠有 Bottom-Up 的外衣，但当你剖开其深层的递归神经图与降维流向走势，它毫无疑问是 <strong>从那高悬天际、代表着独裁全宇宙总汇总口径的最顶端神谕 Apex 节点 (0-D，全局一统) 直接发号施令，劈空斩下</strong>，一步一血印地向下方基层维度空间发动 <strong>降维细化 (Partition)</strong> 的狂暴清剿。</p>
<p><strong>BUC 极其残暴且致命的早停剪枝 (Prune) 绝杀奥义</strong>：<br>
整个 BUC 引擎架构之所有能够在多维组合大爆炸的绝境中傲视群雄，全拜它骨子里死死咬住了那条跟 Apriori 先验法则心法如出一辙的 <strong>单调不可逆降级铁律 (Monotonicity)</strong>。<br>
闭上眼睛冥想那个无可动摇的物理事实边界：任何一个父辈全集巨无霸节点的总人头 <code>Count</code> 储量，必然是呈现出无死角泰山压顶的碾压之势，它绝不可能被其下辖进一步切割衍生出的那些细碎散乱的小格子分号给逆反超越（比如广东省的总编制人口上限卡死在那儿，下面区区一个深圳市无论怎么折腾，它圈到的人都休想大得过广东全局）。<br>
既然如此，在每一次那惊心动魄的自顶向下劈落递归流中，系统只消稍稍探一下鼻息：<br>
<strong>如果就在当前这个正被刀斧加身的半中腰过渡维度的统计口径格子内，它身上攒到的那点可怜巴巴的 Count / Sum 数据血量指标，已经提前拉跨、干瘪微弱到可耻地跌破了最高长官预设的冰山生存最低配额警戒底线 (Iceberg Threshold)——大局已定！它绝不用再去傻乎乎地劳心费神深入翻找了。因为基于那个伟大的单调性禁制结界，它底下以后再怎么继续碎尸万段深挖降维切割产生的所有徒子徒孙末裔格子的继承人，它们分到的可怜遗产绝对不可能起死回生逆向反弹暴涨突破回那个硬性指标生死线上！</strong><br>
伴随这个铁一般的判断斩出，算法引擎根本不带一丝一毫的留恋犹豫，直接就在这里宣判死刑，雷厉风行地拉下那重若千钧的 <strong>连坐整枝隔断大闸 (Prune)</strong>！从这一环向下牵扯出来的整条分支血脉、全部那成千上万原本该去老实排队轮询苦算的无边海量层级维度组合，都在这瞬间的当头棒喝下被毫不留情地连根一脚全盘踹走、被这道神圣裁决尽数挡在算法算力的结算黑名单之外！这种对计算资源的极度血腥碾压和恐怖精简豁免机制，当仁不让地将 BUC 加冕为了应对现代那些令人绝望的超高维极度稀疏恐怖废墟数据集的、不容任何质疑的唯一霸主王座。</p>

<h2>4. CCIC (Closed Cube and Iceberg Cube) 的极致闭环压缩</h2>
<p>单靠冰山截断还不够狠。如果在切分时遇到极其密集的关联属性群，就会诞生一种无语的景象：从“某款特殊游戏机”下钻到了“购买该游戏机的特定骨灰级玩家群体”，发现两者的汇总统计值（Count/Sum）一模一样，毫无流失。<br>
这意味着增加进来的这个所谓新切割维度对这批特定数据完全就是多余的陪衬废话，它根本没有起到实质的再细化切割作用！<br>
<strong>Closed Cube</strong> 技术借用了关联规则中闭合项集的祖传精髓：借助 <strong>Ancestor-descendant (祖先后代父子纽带机制)</strong> 进行扫描，凡是发现这种和它的老一辈子辈算出来的聚合指标一模一样的冗余节点，一律不予重复存储结块，仅仅保留最具宏观代表性的 <strong>闭合单元 (Closed cell)</strong>，再次逼出了压缩空间的极限天花板。其脉络传递流转经常被绘制和使用 <strong>Hass diagram (哈斯图)</strong> 来展示其包含关系。</p>
<h3>🎯 Mock Exam 经典例题</h3>
<p><strong>【Q1. 判定连线题】</strong>如果在某个数据仓库引擎的底层设计中，想要强行跨层级推导出父辈大盘的宏观汇总指标，但面临的数据分别是：总点数 (Count)，平均客单价 (Average)，以及最长消费耗时 (Max)。请你毫不留情地为它们判定其属于何种聚合血统 (Aggregate Function)，并说明是否能借助子树的局部指标计算得出整体结果。</p>
<p><strong>【解答】</strong>:<br>
- <strong>Max (最大值) -> Distributive (分配型)</strong>。直接扫所有小弟的最大值再求最大，绝对无损得出真神。<br>
- <strong>Count (总点数) -> Distributive (分配型)</strong>。简单的人头加总，直接相加即可无脑获解。<br>
- <strong>Average (均值) -> Algebraic (代数型)</strong>。绝对不能两两均值相加除2！而是需要保存上面的 <code>Sum</code> 和 <code>Count</code> 辅助进行代数除法方可解围破关。</p>
<p><strong>【Q2. 算法推理辨析】</strong>BUC 算法明明声称自己叫 Bottom-Up (自底向上)，但它算法运行追踪时明明是从最高处的代表全集的 Apex 节点启动一直往下细分开枝散叶进行钻取 (Drill-down) 的，请问它为何要取这个倒反天罡的迷惑性名字？</p>
<p><strong>【解答】</strong>: 之所以取名自底向上，是因为如果观察最终生成出来的那个实体多维矩阵立方阵（Cuboid网格），它是从拥有维度极少、体积最渺小、但涵盖全局的 Apex 星核（0维）开始算，算好了这颗核心，然后往外蔓延，顺势一点点往外构建起整个越发庞大、涉及维度更深更高、网格密布最广袤的底层海量 Base Cuboid。正是这种网格构建体积上的一点点生长叠加，赋予了它这个极具传奇迷惑色彩的名字。</p>
"""

m14 = """
<h1>MODULE 14: 必考计算题公式全景覆盖</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>综合提取自全部幻灯片</code></p>
<div class="highlight"><strong>专题说明：</strong>汇集历年试卷与PPT中所有带公式的推演逻辑，彻底击穿任何算术陷阱。请在草稿纸上反复默写以下公式。</div>
<h2>1. 混淆矩阵与进阶评估计算</h2>
<ul>
    <li><strong>基石定位：</strong> 行是 Actual (真实)，列是 Predicted (预测)。</li>
    <li><strong>Accuracy:</strong> $\\frac{TP+TN}{TP+TN+FP+FN}$ （全局正确率，在极度不平衡下是谎言）</li>
    <li><strong>Precision (查准率):</strong> $\\frac{TP}{TP+FP}$ （被抓的人里，有几个真贼？警察最关心）</li>
    <li><strong>Recall / TPR (查全率):</strong> $\\frac{TP}{TP+FN}$ （真贼里，我们抓到了几个？癌症检测最关心）</li>
    <li><strong>Specificity / TNR:</strong> $\\frac{TN}{TN+FP}$ （没病的人里，多少被正确排除了？）</li>
    <li><strong>NPV (Negative Predictive Value):</strong> $\\frac{TN}{TN+FN}$ （被判定健康的人里，多少真健康？）</li>
    <li><strong>F-measure:</strong> $2 \\times \\frac{Precision \\times Recall}{Precision + Recall}$</li>
    <li><strong>F-beta:</strong> $F_\\beta = (1 + \\beta^2) \\times \\frac{Precision \\times Recall}{\\beta^2 \\times Precision + Recall}$</li>
</ul>
<h2>2. 决策树纯度指标分裂与剪枝惩罚</h2>
<ul>
    <li><strong>Shannon's Entropy (信息熵):</strong> $H(S) = -\\sum_{i=1}^c p_i \\log_2(p_i)$</li>
    <li><strong>Information Gain:</strong> $IG(S, A) = H(S) - \\sum_{v \\in A} \\frac{|S_v|}{|S|} H(S_v)$</li>
    <li><strong>Split Information:</strong> $SplitInfo(A) = -\\sum_{v \\in A} \\frac{|S_v|}{|S|} \\log_2 \\left( \\frac{|S_v|}{|S|} \\right)$</li>
    <li><strong>Gain Ratio (增益率):</strong> $\\frac{IG(S, A)}{SplitInfo(A)}$ （完美克服信息增益偏爱多分支属性的绝招）</li>
    <li><strong>Gini Impurity:</strong> $Gini(S) = 1 - \\sum_{i=1}^c p_i^2$ （CART树最爱，计算比对数快）</li>
</ul>
<h2>3. K-NN 距离、归一化与权重逆衰减</h2>
<ul>
    <li><strong>Minkowski Distance:</strong> $d(x, y) = \\left( \\sum_{i=1}^n |x_i - y_i|^p \\right)^{\\frac{1}{p}}$ （$p=1$是曼哈顿，$p=2$是欧氏距离，$p \\to \\infty$是切比雪夫）</li>
    <li><strong>Min-Max Normalization:</strong> $v' = \\frac{v - \\min_A}{\\max_A - \\min_A}$ （必须掌握，否则数值大的特征瞬间压垮KNN）</li>
    <li><strong>Z-score Standardization:</strong> $v' = \\frac{v - \\mu_A}{\\sigma_A}$ （抵抗异常离群点）</li>
    <li><strong>Inverse Distance Weighting:</strong> $w_i = \\frac{1}{d(x, x_i)^2}$ 或 $w_i = \\frac{1}{d(x, x_i)}$ （给更近的邻居更高的投票权，避免被远处噪点带偏）</li>
</ul>
<h2>4. AdaBoost 的加权错误率推导</h2>
<ul>
    <li><strong>Error of classifier $m$:</strong> $\\epsilon_m = \\sum_{i=1}^N w_i I(y_i \\neq h_m(x_i))$ （只加总那些被错分样本的权重）</li>
    <li><strong>Classifier Weight $\\alpha_m$:</strong> $\\alpha_m = \\frac{1}{2} \\ln \\left( \\frac{1 - \\epsilon_m}{\\epsilon_m} \\right)$ （错误率越小，树的发言权越大）</li>
    <li><strong>Final Ensemble Voting:</strong> $H(x) = sign \\left( \\sum_{m=1}^M \\alpha_m h_m(x) \\right)$</li>
</ul>
<h2>5. 63.2% Bootstrap 极限概率推演</h2>
<ul>
    <li><strong>单次未命中概率:</strong> $1 - \\frac{1}{N}$</li>
    <li><strong>N 次有放回连抽全未命中:</strong> $(1 - \\frac{1}{N})^N$</li>
    <li><strong>极限值 (Out-of-Bag比例):</strong> $\\lim_{N \\to \\infty} \\left(1 - \\frac{1}{N}\\right)^N = \\frac{1}{e} \\approx 36.8\\%$</li>
    <li><strong>抽中比例:</strong> $1 - 36.8\\% = 63.2\\%$ （训练集实际只包含63.2%的独特数据）</li>
</ul>
<h2>6. Apriori 的频繁模式计算</h2>
<ul>
    <li><strong>Support:</strong> $\\frac{Count(X \\cup Y)}{Total\\ Transactions}$</li>
    <li><strong>Confidence:</strong> $\\frac{Support(X \\cup Y)}{Support(X)}$ （在买了X的人里，有多少顺便买了Y？）</li>
    <li><strong>Lift:</strong> $\\frac{P(X \\cup Y)}{P(X) \\times P(Y)}$ （大于1才代表真实的正向促进，完美识破高置信度的假象）</li>
</ul>
<h2>7. 聚类分析计算矩阵</h2>
<ul>
    <li><strong>K-Means 优化目标 (SSE):</strong> $SSE = \\sum_{i=1}^K \\sum_{x \\in C_i} ||x - \\mu_i||^2$</li>
    <li><strong>层次聚类 Single-Linkage:</strong> $d(A, B) = \\min_{x \\in A, y \\in B} d(x, y)$</li>
    <li><strong>层次聚类 Complete-Linkage:</strong> $d(A, B) = \\max_{x \\in A, y \\in B} d(x, y)$</li>
    <li><strong>OPTICS Core-distance:</strong> 满足 $\\ge MinPts$ 邻居所需的最短半径</li>
    <li><strong>OPTICS Reachability-distance:</strong> $\\max(Core\\_distance(o), d(p, o))$</li>
    <li><strong>Silhouette Coefficient (轮廓系数):</strong> $s(i) = \\frac{b(i) - a(i)}{\\max(a(i), b(i))}$ （结果介于 $[-1, 1]$，越接近1越说明内聚外斥极其完美）</li>
</ul>
"""

m15 = """
<h1>MODULE 15: WEKA常见算法与伪代码深度推演</h1>
<p style="font-size: 14px; color: var(--muted); border-left: 4px solid var(--border); padding-left: 12px; margin-top: -16px; margin-bottom: 32px;">📄 <strong>参考讲义：</strong><code>综合提取自全部幻灯片</code></p>
<div class="highlight"><strong>专题说明：</strong>针对可能出现的算法“执行步骤排序题”与 WEKA 算法填空题进行针对性梳理。所有算法均已采用 WEKA 官方实现命名并内置融合到各个前序基础知识模块内部中去（即您在浏览诸如 M03, M04, M08 等独立知识图鉴时，其相关的 WEKA 工具具体执行流与拦截代码便会赫然展现）。此处汇聚做一集中快速索引。</div>
<h2>1. 分类兵器库大观 (Classification Flows)</h2>
<p>本课程重火力压制的决策高塔与判决铁链：</p>
<ul>
    <li><strong>决策树 J48 (C4.5) 引擎：</strong> 凭借 Gain Ratio 规避多值诱惑，在叶节点纯净度不足或样本微缩时强行收手。事毕引爆 <code>confidenceFactor</code> 开启暴戾的后方大清算剪枝。详情点击参考 <strong>[M03 模块]</strong>。</li>
    <li><strong>决策树 REPTree 后方防御：</strong> 最简陋的建树流程，配以最精明的分裂自保——直接祭出 Holdout 验证集，启动 Reduced Error Pruning 机制，发现验证集上犯错稍高，立刻截断，护身保命第一。详情点击参考 <strong>[M03 模块]</strong>。</li>
    <li><strong>纯血规则集 JRip 引擎 (RIPPER)：</strong> 永远采用 Separate-and-Conquer （摘一叶杀一人的分离并征服大法）抽出高精专防规则。其底牌为祭出 64bits 的 MDL 死亡惩罚线拦腰叫停疯狂的规则生长线。详情点击参考 <strong>[M05 模块]</strong>。</li>
    <li><strong>屠杀成树法则 PART：</strong> 最血腥的提取机器，直接现场建起一颗巨大的 C4.5 大树，蛮力剥下最为粗壮、笼盖面极广的那一条金叶规则。得手后立刻翻脸无情，摧毁抛弃全树。详情点击参考 <strong>[M05 模块]</strong>。</li>
    <li><strong>零帧起手式 ZeroR：</strong> 数据界的最低门槛废铁标尺，毫无预测可言，直接无脑盲猜所有样本里人头数最庞大的多数派（Majority Class），作为检验高级魔法有无翻车的绝对下限防御墙。详情点击参考 <strong>[M02 与 M05 模块]</strong>。</li>
    <li><strong>近邻巡航法 IBk (KNN)：</strong> 极其纯粹的空间扫描探测算法，无脑吃掉内存存下所有点。测试集一来，立刻拉平量纲（Min-Max/Z-score），打开欧氏尺暴力扫描。并且内置 <code>distanceWeighting</code> 的距离偏袒加权绝技，把投票权极端倒向贴身心腹，瞬间瓦解外围异类噪点干扰。详情点击参考 <strong>[M04 模块]</strong>。</li>
    <li><strong>极寒方差收割机 Random Forest：</strong> Bagging 流派的最高造诣结晶。除了拉上成千上万多胞胎分身去狂抽大投票外，更强行挖除了建树节点的全盘上帝视野，把特征选择权完全禁锢在一个可怜的窄口切片框里 ($K = \\log_2M+1$) 逼着系统畸变生长，最后换来了这堪称奇迹般无比多样的集成抗压性。详情点击参考 <strong>[M06 模块]</strong>。</li>
    <li><strong>孤注一掷逆风盘 AdaBoost：</strong> Boosting 流派的极限接力救援体系。永远强迫后人全盘承接过往兄弟失败造就的重压烂摊子——那批被不断滚雪球般暴力提高难度权重的高频错分死结题。逼出强人后，根据在训练时的胜率分配独裁发言权，完成逆转乾坤的一波强效非线性拟合。详情点击参考 <strong>[M06 模块]</strong>。</li>
</ul>

<h2>2. 无监督混沌聚类法 (Unsupervised Clustering Flows)</h2>
<p>在缺乏上帝真实标签的黑暗地带，算法自我救赎的法则：</p>
<ul>
    <li><strong>质心强权引力法 SimpleKMeans：</strong> 开场随机盲投引爆极高风险。后利用空间大挪移，一刻不停地执行 Assignment(收拢小弟) 和 Update(移交新质心位置) 两步交接曲舞。直到整体混沌差值 $SSE$ 的波谷见底、小弟队伍的军心不再动摇，算法强制宣告停摆固化。详情点击参考 <strong>[M09 模块]</strong>。</li>
    <li><strong>层次滚雪球法 (Hierarchical AGNES)：</strong> 不看全局，只在浩渺中挑选一对最来电的基建结拜。利用 Single/Complete/Ward 各种连线绝活拼凑树枝，一旦绑定生生世世绝不回头反悔，连绵不断长出那巨大的分岔系统进化树大合影图。详情点击参考 <strong>[M09 模块]</strong>。</li>
    <li><strong>反骨抗噪雷达法 DBSCAN：</strong> 从未被球形束缚的自由魂。用 $Eps$ 和 $MinPts$ 扫描划分王侯阶级（核心）、边缘炮灰（边界）与被放逐者（噪声）。借由核心传销般的密度可达网脉向四周无节制辐射勾连，直至拼合出一只极度畸形却又紧密团结的怪兽级簇群。详情点击参考 <strong>[M10 模块]</strong>。</li>
</ul>

<h2>3. 黄金屋商海淘宝战 (Frequent Patterns & Data Warehouse)</h2>
<p>打破组合诅咒与极速查账的引擎心法：</p>
<ul>
    <li><strong>组合砍伐绝学 Apriori：</strong> 把组合大爆炸玩弄于股掌。通过 $L_{k-1}$ 自我联姻诞生出 $C_k$ 大批候选。紧接着祭出最引以为傲的“反单调性断头台”：子集里但凡有一个没资格当纯血贵族的，这个侯选项被立刻拉去处决。经过极其惨烈的精简屠杀后才去数据库里读盘数钱验证，完美破除高维检索时间黑洞。详情点击参考 <strong>[M08 模块]</strong>。</li>
    <li><strong>数据立方体的绝路审判 BUC (Bottom-Up Construction)：</strong> 身披自底向上的名号，却干着自高台极速俯冲切割勾当的神技。这把快刀一旦下落探入格子，只要发现 Count 这点血量微弱到不满足冰山警戒线的极低门槛，连眼皮都不抬一下，当场宣判此脉斩绝，它底下所有本该接连被精雕细切的子子孙孙切片网格维度瞬间被强行全盘阻断剥夺算力名额，造就了多维分析运算史上最恐怖的无损削减传奇。详情点击参考 <strong>[M13 模块]</strong>。</li>
</ul>
"""

# Extract the data directly to a data.js file
import json
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
