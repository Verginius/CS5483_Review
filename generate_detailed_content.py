import os
import json

m01 = """
<h1>MODULE 01: 概述与数据预处理</h1>
<div class="highlight"><strong>核心考点：</strong>KDD流程全貌、数据特征的具体类型、缺失值与冗余特征处理、距离与相似度度量深度分析。</div>
<h2>1. 什么是数据挖掘与知识发现 (KDD)?</h2>
<p>数据挖掘 (Data Mining) 是知识发现 (Knowledge Discovery from Databases, KDD) 过程中最为核心的一环。严格按照课件原图，完整的 KDD 包含以下 <strong>4 个核心步骤</strong>：</p>
<ol>
    <li><strong>Preparation</strong>
        <ul>
            <li>set the goal (what to learn)</li>
        </ul>
    </li>
    <li><strong>Pre-processing</strong>
        <ul>
            <li>Data cleaning/integration: handle noise/errors/missing values</li>
            <li>Data selection/reduction/transformation/discretization: create target data set with relevant samples/variables</li>
        </ul>
    </li>
    <li><strong>Data mining</strong>
        <ul>
            <li>Apply learning algorithm(s) to compute desired patterns from processed data</li>
        </ul>
    </li>
    <li><strong>Interpretation</strong>
        <ul>
            <li>Iterate if performance is unsatisfactory</li>
            <li>Deploy if ready: report, incorporate, apply</li>
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
<h2>3. 距离与相似度度量详解</h2>
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
<p><strong>【解答】</strong>: 完整的 KDD 步骤包含 4 大阶段：<br>1. Preparation<br>2. Pre-processing<br>3. Data mining<br>4. Interpretation<br>其中，部署 (Deploy if ready: report, incorporate, apply) 被归类在第四个核心大阶段 <strong>Interpretation</strong> 当中。</p>
<p><strong>【Q2. 计算题】</strong>给出两个向量 $X = (1, 3)$ 和 $Y = (4, 7)$，请分别计算它们之间的曼哈顿距离、欧氏距离和切比雪夫距离。</p>
<p><strong>【解答】</strong>:<br>
- 曼哈顿距离 ($L_1$): $|1-4| + |3-7| = 3 + 4 = 7$<br>
- 欧氏距离 ($L_2$): $\\sqrt{(1-4)^2 + (3-7)^2} = \\sqrt{9+16} = 5$<br>
- 切比雪夫距离 ($L_\\infty$): $\\max(|1-4|, |3-7|) = \\max(3, 4) = 4$</p>
"""

m02 = """
<h1>MODULE 02: 分类模型性能评估</h1>
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
<h2>4. ROC 曲线与 PRC 曲线深度对比</h2>
<p>通过 <strong>Threshold-moving (移动判定阈值)</strong>，每个算法可以产出一系列 TP 和 FP 的组合。在 WEKA 中，可以通过 ThresholdCurve 观察这些动态特性。</p>
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
<h2>3. 剪枝策略 (Pruning Strategies) 防过拟合</h2>
<ul>
    <li><strong>Pre-pruning (预剪枝 / Early Stopping)</strong>：在树生长时限制它。比如设定最大深度、设定分裂最小信息增益阈值、或 WEKA J48 中的 <code>minNumObj</code> (每个叶子节点必须包含的最少实例数)。预剪枝极快，但容易因过早停止而造成欠拟合。</li>
    <li><strong>Post-pruning (后剪枝)</strong>：树长到最大规模后，自底向上评估。如果将子树折叠为单一叶节点所带来的误差惩罚（在验证集上）小于模型复杂度惩罚，则进行裁剪。经典方法如 CART 的代价复杂度剪枝。后剪枝效果更好，但计算极其昂贵。</li>
</ul>
<h2>4. WEKA 中常用的树模型对比 (J48 vs REPTree vs RandomTree)</h2>
<table border="1" style="width:100%; border-collapse: collapse; margin-bottom: 20px;">
    <tr><th>WEKA 算法</th><th>原算法对应</th><th>核心特性与参数</th><th>适用场景与优缺点</th></tr>
    <tr>
        <td><strong>J48</strong></td>
        <td>C4.5</td>
        <td>基于 Gain Ratio。内置了后剪枝机制。关键参数 <code>confidenceFactor</code> ($C$，决定剪枝严格程度，默认0.25，越小剪枝越狠) 和 <code>minNumObj</code> ($M$，叶节点最少样本)。</td>
        <td>最通用的决策树，可解释性极强，支持名义与数值属性。但在高噪、高相关性特征时容易过拟合。</td>
    </tr>
    <tr>
        <td><strong>REPTree</strong></td>
        <td>Reduced Error Pruning Tree</td>
        <td>基于信息增益或方差（可做回归）。使用 Holdout (保留出独立验证集) 专门进行快速的后剪枝 (Reduced Error Pruning)。</td>
        <td>构建速度极快，防过拟合能力在强噪声数据上比 J48 更有效，但由于切出了独立验证集，对小样本数据集极不友好。</td>
    </tr>
    <tr>
        <td><strong>RandomTree</strong></td>
        <td>-</td>
        <td>在分裂时不考察所有属性，而是只考察随机抽取的 $K$ 个属性。不剪枝。</td>
        <td>单树精度低，极易过拟合，通常绝不单独使用，而是专门作为随机森林 (Random Forest) 的基础构建块。</td>
    </tr>
</table>

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
<h2>5. WEKA 中的 KNN 实现：IBk (Instance-Based learning with k)</h2>
<p>在 WEKA 中，KNN 对应的算法是 <strong>IBk</strong> (位于 <code>lazy</code> 分类下)。</p>
<ul>
    <li><strong>核心参数 <code>k</code></strong>：设定近邻数，默认通常是 1。可以使用 Cross-validation 自动选择最佳的 $K$（通过设置 <code>crossValidate</code> 为 True）。</li>
    <li><strong>参数 <code>distanceWeighting</code></strong>：默认是对前 $K$ 个邻居一视同仁。但如果设置距离倒数加权 (Inverse Distance)，那么虽然选了 5 个邻居，但距离你越近的那个邻居拥有越大的投票权：$w_i = \\frac{1}{d(x, x_i)}$。这能够极大削弱孤立噪点带来的负面干扰。</li>
    <li><strong>时间换空间问题</strong>：为了解决查询慢的问题，WEKA 中的 IBk 允许配置底层的查找结构（如 <code>LinearNNSearch</code>, <code>KDTree</code>, <code>BallTree</code>）来将搜索复杂度降至 $O(\\log N)$。</li>
</ul>

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
<div class="highlight"><strong>核心考点：</strong>规则提取的极高解释性、Sequential Covering (分离并征服) 与分而治之的本质区别、RIPPER防止过拟合的 MDL 原则、WEKA中 JRip、PART的运作细节。</div>
<h2>1. 基于规则分类的不可替代性</h2>
<p>即使在神经网络(Black-box)大行其道的今天，Rule-Based 系统在金融风控、医疗拒保系统、法律定罪等领域依然是王者。因为一组精简的 <strong>IF-THEN 规则</strong> 具备人类可以直接阅读、审计、并强行进行局部删改的特点（即 <strong>Modular 模块化</strong> 与 <strong>Interpretable 解释性</strong>）。一棵庞大的决策树如果改动一个上层节点，整棵树都必须重构，而规则集可以直接禁用某一条不良规则。</p>
<h2>2. 生成规则的两大范式</h2>
<ul>
    <li><strong>1. 间接生成 (Extract rules from Decision Trees)</strong>：<br>
    使用 C4.5 生成一棵树，然后每一条从 Root 走到 Leaf 的路径就是一条巨长无比的规则。随后使用算法将路径上的无用判定条件修剪掉。</li>
    <li><strong>2. 直接生成 (Sequential Covering / 顺序覆盖)</strong>：<br>
    这就是著名的 <strong>Separate-and-conquer (分离并征服)</strong>。
    <ol>
        <li>在所有数据上，学习/生成出一条纯度、覆盖度最好的一条单独规则。</li>
        <li><strong>Separate</strong>：将所有被这条规则正确捕捉到的实例，从训练集中彻底物理删除！</li>
        <li><strong>Conquer</strong>：在剩余越来越少的数据集上继续学习下一条规则，直到数据全被覆盖。</li>
    </ol>
    这种方法与决策树的 Divide-and-conquer 最大的不同在于：决策树是在拆分子集，所有子集拼起来是一整棵树；而 Sequential covering 是学一条好规则，切走一块蛋糕，再学下一条。</li>
</ul>
<h2>3. 规则的覆盖冲突 (Conflict Resolution)</h2>
<p>如果新来一个用户，他同时触发了 Rule 1 (判死刑) 和 Rule 2 (判无罪)，怎么办？</p>
<ul>
    <li><strong>Size ordering</strong>：优先相信触发条件苛刻（前件更长更具体）的规则。</li>
    <li><strong>Class-based ordering</strong>：按类别的罕见性排序（比如优先判别为少数派欺诈类别）。</li>
    <li><strong>Rule-based ordering (Decision List)</strong>：将规则严格按优先级排成一个长长的 List。测试实例顺着 List 往下走，触发了哪条就立刻输出结果并停止验证后面的规则。WEKA 中的 JRip 产出的就是这种 Decision List。</li>
</ul>
<h2>4. WEKA 中的 Rule-Based 算法分析</h2>
<table border="1" style="width:100%; border-collapse: collapse; margin-bottom: 20px;">
    <tr><th>WEKA 算法</th><th>代表思想</th><th>运作原理与防止过拟合机制</th><th>优缺点</th></tr>
    <tr>
        <td><strong>PART</strong></td>
        <td>结合 C4.5 的“部分决策树”</td>
        <td>PART 也是走 Sequential Covering 流程，但是它在找“最好的单条规则”时，会在当前数据上建一棵 <strong>局部 C4.5 决策树</strong>。找到那条覆盖面积最广（最好）的叶子规则后，立刻抛弃整棵树。剔除对应数据，循环上述过程。</td>
        <td>优点是不需要进行复杂的后续规则优化阶段 (Global optimization)，提取的规则极其纯净准确。缺点是反复造树再抛弃，计算量庞大。</td>
    </tr>
    <tr>
        <td><strong>JRip (RIPPER)</strong></td>
        <td>纯粹的增量式规则归纳</td>
        <td>采用 <strong>FOIL (First Order Inductive Learner) 信息增益</strong>来生长规则。<br>
        <strong>如何防过拟合？</strong> 使用 <strong>MDL (Minimum Description Length, 最小描述长度)</strong> 理论。系统实时监控新规则的描述长度（即模型复杂度），如果新增一条规则带来的正确率提升不足以抵消它增加的超过 64 bits 的冗长描述，算法立即终止。最后进行一系列替换和全局优化。</td>
        <td>规则极其简洁，泛化能力极强，是直接生成规则的最先进算法之一。但在极其非线性的数据上，表达能力弱于复杂树模型。</td>
    </tr>
    <tr>
        <td><strong>ZeroR</strong></td>
        <td>Baseline (基线模型)</td>
        <td>无视一切特征，直接猜测数据集中出现次数最多的那个类别。</td>
        <td>绝对的最差基准，毫无准确率可言。如果任何复杂算法（如 JRip 或 J48）的准确率还不如 ZeroR，说明算法不仅没学到东西，还把数据搞砸了。</td>
    </tr>
</table>

<h3>🎯 Mock Exam 经典例题</h3>
<p><strong>【Q1. 算法执行流程考点】</strong>请阐述 JRip（WEKA中的 RIPPER 算法）是依靠什么底层的停止机制来防止提取出来的规则集过于庞杂、死记硬背而导致过拟合的？</p>
<p><strong>【解答】</strong>: JRip 依靠 <strong>MDL (Minimum Description Length, 最小描述长度)</strong> 原则作为它的核心停止机制 (Stopping criterion)。在执行 Sequential Covering (逐步增加新规则) 或扩充某条规则前件时，算法会实时监控这套规则在理论上的“位宽/信息体积”。如果最新加入的一条规则虽然略微提高了训练集的纯净度，但它带来的模型长度扩充超过了容忍的惩罚阈值（通常设定为 64 bits），系统就会认定得不偿失，立刻叫停该规则的生长以保持模型的泛化能力。</p>

<p><strong>【Q2. 概念对比】</strong>PART 和 JRip 虽然都是用来挖掘独立规则集的，请问它们在生成“最好的一条规则”时的策略有何本质区别？</p>
<p><strong>【解答】</strong>: JRip 采用的是增量生长策略，每次挑一个最优特征（通过 FOIL 信息增益）直接接上 <code>AND</code> 条件；而 PART 的做法极度暴力——它利用当前所有未被覆盖的实例，直接调用类似 C4.5 的建树算法去强制生长出一棵“残缺版的局部决策树 (Partial Tree)”，然后挑这棵树上看起来覆盖面积最庞大、纯度最好的叶子节点路径强行拔下来作为提取到的第一条规则，然后立刻把剩下的树丢进垃圾桶。这种“局部造树”赋予了它极高的准确性，但计算损耗极大。</p>
"""

m06 = """
<h1>MODULE 06: Ensemble Methods (集成学习)</h1>
<div class="highlight"><strong>核心考点：</strong>Bias-Variance 理论下的偏差方差权衡、Bagging的Bootstrap抽样平滑方差机制、Random Forest特征随机化、Boosting(AdaBoost)的加权纠错降低偏差机制。</div>
<h2>1. 为什么要使用集成方法？(Ensemble Philosophy)</h2>
<p>单一的分类器往往具有不可逾越的瓶颈：浅层决策树虽然稳健但 <strong>偏差 (Bias)</strong> 极高，只能划出四四方方的决策块（欠拟合）；深层决策树虽然拟合力强，但极其容易被局部噪声诱导，<strong>方差 (Variance)</strong> 极高，泛化一塌糊涂。<br>
<strong>Ensemble (集成)</strong> 的哲学是：将多个相互独立且具有 <strong>多样性 (Diverse)</strong> 的弱基分类器结合起来，能够平滑噪声降低方差，或串联纠错降低偏差，从而打破单一算法的性能天花板。这也是所有 Kaggle 竞赛中的屠榜利器。</p>
<h2>2. Bagging (Bootstrap Aggregation - 装袋法)</h2>
<p>Bagging 的核心目标是：<strong>拯救那些方差极高、极易过拟合的算法（比如不剪枝的深层决策树、KNN）</strong>。通过多数人的力量把不稳定的预测给“拉平”。</p>
<ul>
    <li><strong>工作原理</strong>：采用 <strong>Bootstrap (有放回随机抽样)</strong>。如果原数据集有 N 个样本，每次都有放回地抽 N 次组成一个新数据集。因为是有放回，每个新数据集里都会有重复数据，同时有大约 36.8% 的原数据（Out-of-Bag, OOB 数据）根本没被抽到。用这 M 个稍微不同的新数据集，独立训练出 M 个基分类器。</li>
    <li><strong>融合策略</strong>：由于 M 个分类器完全独立（可以并行计算），对于分类问题，进行 <strong>多数投票 (Majority Voting)</strong>；对于回归问题，进行求平均。</li>
    <li><strong>结论</strong>：由于各个基模型是独立的，求平均后方差极大地降低了。如果个别树对噪声过拟合，投票时它们会被其他正常的树淹没掉。</li>
</ul>
<h2>3. Random Forest (随机森林) - WEKA 中的绝对主力</h2>
<p>随机森林是 Bagging 的终极形态。它敏锐地发现：即便使用了 Bootstrap 数据，但如果数据里有几个强相关的核心特征，那么长出来的 M 棵树还是会长得一模一样，缺乏最重要的 <strong>多样性 (Diversity)</strong>。<br>
<strong>RF 的突破性操作</strong>：在建每一棵树的过程中，当要在某个节点寻找最佳切分属性时，算法 <strong>不看所有的属性</strong>，而是 <strong>随机挑出一个包含 K 个特征的子集</strong>（通常 $K = \\log_2(总特征数)$），只在这个狭窄的子集里找最优分割！这强制逼迫原本长相相同的树走上不同的发展路径，彻底打断了树与树之间的相关性。</p>
<h2>4. Boosting (提升法 - AdaBoost)</h2>
<p>如果说 Bagging 是为了降方差，那么 Boosting 的核心目标是 <strong>拯救那些偏差极高、根本学不会复杂模式的“弱智”分类器（如 Decision Stump, 决策树桩，只裂开一层的树）</strong>，即降低偏差 (Reduce Bias)。</p>
<ul>
    <li><strong>串行纠错原理</strong>：Boosting 无法并行。它训练好第一棵基分类器后，去检查哪些样本被分错了。在训练第二棵分类器时，系统会 <strong>人为地给那些第一棵树分错的困难样本施加极高的权重 (Weights)</strong>。这意味着第二棵树会被迫专注于纠正前人的错误。依次类推，第 M 棵树专治前面所有人的疑难杂症。</li>
    <li><strong>AdaBoost (Adaptive Boosting)</strong>：最著名的 Boosting 算法。在最后整合所有的弱分类器时，它不会使用民主平等的多数投票。而是给在训练集上总错误率低的基分类器极高的 <strong>发言权权重</strong>，给经常出错的分类器极低的权重。进行 <strong>加权多数投票 (Weighted Majority Voting)</strong>：最终预测为 $H(x) = sign\\left(\\sum_{m=1}^M \\alpha_m h_m(x)\\right)$，其中 $\\alpha_m$ 为分类器的权重。</li>
</ul>
<h2>5. Stacking (堆叠)</h2>
<p>Stacking 是一种更为高阶的融合法。在 WEKA 中，你可以把 J48、KNN、Logistic Regression 分别跑一遍，它们各自会给出一个预测结果（如分类概率）。Stacking 把这些基础算法输出的预测概率 <strong>当作新的输入特征</strong>，喂给最高层的一个 <strong>元分类器 (Meta Classifier)</strong>，让元分类器自己去学习到底该相信谁在什么情况下的判断。这是一种机器自己学习如何融合结果的黑科技。</p>

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
<div class="highlight"><strong>核心考点：</strong>深刻理解泛化与过拟合的本质对立，经验风险最小化(ERM)理论，以及交叉验证、63.2 Bootstrap的数据切分数学原理与应用场景。</div>
<h2>1. 问题的本质：Generalization (泛化) vs Overfitting (过拟合)</h2>
<p>我们训练模型的终极目的，绝对不是让模型去记住历史数据，而是为了应对 <strong>未来全新的、未知的数据 (Unseen / Future data)</strong>。这就叫做模型的 <strong>Generalization (泛化能力)</strong>。<br>
<strong>Overfitting (过拟合)</strong> 指的是：当模型拥有极高的复杂度（参数过多、树极深）时，它能够像背诵课文一样，把训练集 (Fitted data) 中的特异性噪声、巧合、甚至是离群点完美地记下来，导致训练准确率高达 100%。但只要遭遇没有见过的新样本，它的性能会发生断崖式的崩溃。</p>
<h2>2. 经验风险最小化 (Empirical Risk Minimization, ERM) 的两难</h2>
<p>理想情况下，我们应该基于自然界万物的“真实联合概率分布”去寻找能够让“期望风险 (Expected Risk)”最小的分类器。<br>
但在现实世界中，真实的整体分布是不可知的，我们手中只有几百上千条的 <strong>有限样本抽样</strong>。因此所有机器学习算法退而求其次，只能去最小化这些已知样本上的错误率，即 <strong>经验风险最小化 (ERM)</strong>。<br>
这就是过拟合诞生的根本原因：由于 ERM 是在迎合那部分有限的样本均值，如果你的算法过于贪婪或缺乏正则化约束，它就会将这部分有偏差的局部样本特征当作宇宙真理去拟合。</p>
<h2>3. 估算真实性能的方法大盘点 (Performance Estimation)</h2>
<p>既然不能用训练集的准确率骗自己，我们必须切出独立的数据来检验模型：</p>
<ul>
    <li><strong>Holdout Method (保留法 / 切分法)</strong>：直接将数据按照 7:3 的比例随机切分为训练集和测试集（或加入验证集 Validation set 用于调参）。<br>
        <strong>劣势</strong>：一刀切的方式极具偶然性。如果“大乐透”式地把最难辨认的样本全分进了测试集，模型会被低估；如果测试集全是简单题，模型会被高估。</li>
    <li><strong>Stratification (分层抽样)</strong>：这是极其关键的改进。在 Holdout 时，不是盲目纯随机切，而是强制保证 <strong>测试集和训练集中的类别比例必须与原始数据集完全一致</strong>。例如，原始数据里肿瘤阳性占 5%，那么抽出来的训练集和测试集中，阳性比例也必须严格卡死在 5%。在 WEKA 的评估中，分层是默认开启的核心操作。</li>
    <li><strong>Cross-Validation (交叉验证 / K-Fold CV)</strong>：将数据分成 $K$ 个不重叠的等份 (Folds)。循环 $K$ 次，每次轮流挑出第 $i$ 份作为测试集，剩下 $K-1$ 份合并作训练集。最后把 $K$ 次测试结果求平均。这是目前学术界公认最严谨、方差最小的评估方式。通常采用 10-Fold CV。</li>
</ul>
<h2>4. 63.2 Bootstrap 的数学机理</h2>
<p>当数据集 <strong>极其稀少</strong> (只有几十条数据) 时，我们根本舍不得划出 1/3 去做不能参与训练的测试集。此时使用 <strong>Bootstrap</strong>。<br>
它通过 $N$ 次 <strong>有放回的随机抽样 (Sampling with replacement)</strong> 构造出一个包含 $N$ 条数据的新训练集。</p>
<p><strong>极其高频考点推导</strong>：<br>
在有放回抽样中，某一条特定样本在 1 次抽取中 <strong>没有被抽中</strong> 的概率是 $\\left(1 - \\frac{1}{N}\\right)$。<br>
那么在 $N$ 次连抽中，它始终没有被抽进训练集的概率是 $\\left(1 - \\frac{1}{N}\\right)^N$。<br>
根据高等数学极限，当 $N$ 趋于无穷大时，概率趋近于 $\\lim_{N \\to \\infty} \\left(1 - \\frac{1}{N}\\right)^N = \\frac{1}{e} \\approx 0.368$。<br>
这意味着，构造出来的训练集中，其实只含有大约 <strong>63.2% 的不重复原始样本</strong>！而剩下那 <strong>36.8% 被天生过滤掉的、没有被污染的原生样本 (Out-of-Bag 样本)</strong>，就完美顺理成章地被拿来充当纯净的 Test Set，从而在极端小样本的情况下榨取最大化的数据评估价值。</p>

<h3>🎯 Mock Exam 经典例题</h3>
<p><strong>【Q1. 计算推演题】</strong>某医疗研究机构仅收集到了 100 份罕见病样本进行数据挖掘。如果采用传统 Holdout，测试集过小会导致评估严重失真，于是他们决定采用 63.2% Bootstrap 策略。请问他们必须在原库中执行多少次“有放回抽样”来组建训练集？在这些生成的训练集中，预期会有多少条其实是互不相同的独特样本？剩下的那部分原本没被抽中的孤立样本被拿去做什么用？</p>
<p><strong>【解答】</strong>:<br>
- 必须执行与原样本数相等次数的有放回抽样，即 <strong>100次</strong>。<br>
- 根据极限公式，在经过大量的有放回抽样后，预期有约 $63.2\\%$ 的独特样本进入训练集。所以训练集中大约包含 <strong>63 条互不相同的独立样本</strong>（其余 37 条是由这些独特样本随机重复而拼凑成满 100 条的池子）。<br>
- 剩下的那部分（约 36.8%）始终没有任何一次被抽中、未被污染的原生数据被称为 <strong>Out-of-Bag (OOB) 样本</strong>，它们将被直接充当绝佳的无偏 <strong>Test Set (测试集)</strong>，用来评估该次训练出来的模型性能！</p>
"""

m08 = """
<h1>MODULE 08: Frequent Pattern Analysis (Apriori Algorithm)</h1>
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
<h2>3. Apriori 的 Join & Prune (连接与剪枝算法步骤)</h2>
<p>Apriori 是一个逐层搜索迭代的算法，寻找 $k$ 维频繁项集：</p>
<ol>
    <li><strong>Scan 1</strong>：第一次扫描全库，统计出所有单一商品出现的次数，剔除不足阈值的，留下 <strong>频繁 1-项集 ($L_1$)</strong>。</li>
    <li><strong>Join (连接生成候选)</strong>：使用现有的频繁 $k-1$ 项集 ($L_{k-1}$) 互相连接，生成下一维度的 <strong>候选 $k$-项集 ($C_k$)</strong>。</li>
    <li><strong>Prune (基于先验性质剪枝)</strong>：极其关键的一步！在 $C_k$ 集合中，遍历每个候选项的所有 $k-1$ 维子集。只要发现它的 <strong>任何一个子集</strong> 不存在于之前的频繁集 $L_{k-1}$ 中，立刻将这个侯选项从 $C_k$ 中删除。</li>
    <li><strong>Filter & Count</strong>：对留存的精英 $C_k$ 再次去扫描真实数据库，计算真实的支持度计数。达标的正式成为 $L_k$。</li>
    <li>重复，直到某一代再也产生不了候选项为止。</li>
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
<div class="highlight"><strong>核心考点：</strong>聚类思想的根本区别、K-Means质心更新推导及其面对非凸形状与异常点的脆弱性、三种层次聚合的 Linkage 计算差异与不可逆缺陷、WEKA 中的具体实现探讨。</div>
<h2>1. 聚类分析背景与划分流派 (Partitioning Methods)</h2>
<p>分类算法拥有预先标记的 Ground Truth，而聚类 (Clustering) 是纯粹的无监督学习，它盲人摸象式地试图根据对象自身的内部属性（如几何距离），将数据集分割为 $K$ 个组，使得 <strong>簇内高度内聚，簇间极其疏远</strong>。</p>
<h3>K-Means 算法运转逻辑：</h3>
<ol>
    <li>在茫茫数据海中随机空投 $K$ 个质心 (Centroids)。</li>
    <li>计算每一个样本到这 $K$ 个质心的欧氏距离，把它强行分配给最近的质心领地。</li>
    <li>根据刚刚分配过来的这批小弟，重新计算它们的坐标平均值 $\\mu_i = \\frac{1}{|C_i|} \\sum_{x \\in C_i} x$，将其作为新的质心坐标。</li>
    <li>质心位置变动，原本边界上的一些小弟发现别的质心离自己更近了，导致下一轮领地划分重组。不断迭代，直到所有质心被小弟们固定住，再也无法移动（$SSE = \\sum_{i=1}^K \\sum_{x \\in C_i} ||x - \\mu_i||^2$ 方差和极小值收敛）。</li>
</ol>
<h3>K-Means 的四大致命死穴 (Limitations) 考点：</h3>
<ul>
    <li><strong>对初始落点极度敏感</strong>：纯随机初始化很容易让两个质心降落在同一个自然簇中，导致它陷入毫无意义的 <strong>局部最优解 (Local Optimum)</strong>。在 WEKA 中，SimpleKMeans 往往需要配合 K-Means++ 的聪明初始化或进行多次随机重启取最优。</li>
    <li><strong>无法捕捉非球形结构 (Non-convex bias)</strong>：K-Means 的距离分配和均值运算，使其潜意识里认定簇的形状必须是完美的球体/凸形。面对现实世界中环套环、S形、或两条平行线形态的簇，它会直接一刀切碎，彻底崩坏。</li>
    <li><strong>大小悬殊与同心质心崩溃</strong>：如果数据中一个小簇被包裹在一个大簇的同一个引力中心周围（如同心圆），或者两簇大小相差巨大，它会强行切分导致失真。</li>
    <li><strong>对离群噪声 (Outliers) 毫无抵抗力</strong>：一个处于几千公里之外的错误噪点被强行归入某簇后，在计算均值时会像杠杆一样极其剧烈地将整个质心拉扯向深渊。<br>
    <strong>应对方案：K-Medoids 算法 (PAM)</strong>，它不允许计算虚空的均值质心，而是强制挑选当前簇内最居中的那个 <strong>真实的实体样本</strong> 作为中心。这就好比中位数对抗平均数，极大压制了离群噪点的杠杆效应。</li>
</ul>
<h2>2. 层次流派 (Hierarchical Clustering)</h2>
<p>划分聚类生硬地将数据拍成扁平的 K 块，而层次聚类无需你提前告诉它 K 是多少。它像堆雪人一样，自底向上 <strong>凝聚 (Agglomerative, AGNES)</strong>，最终生成一棵如进化树一般的层次树 (Dendrogram)。你可以像切洋葱一样在任何高度横向切一刀，得到任意数量的簇划分。</p>
<h3>核心考点：决定两个多边形簇之间“距离”的三大几何判定法：</h3>
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
<div class="highlight"><strong>核心考点：</strong>DBSCAN 算法中三大实体（核心、边界、噪声）的严格定义、密度直达与相连的连通性、基于密度如何完美规避 K-Means 的形状限制、以及为了应对变密度问题而进化出的 OPTICS 可达性图技术。</div>
<h2>1. 打破球形迷信：密度聚类的哲学动机</h2>
<p>K-Means 只认同以几何质心为圆心的完美球状簇，层次 Single-linkage 则容易被随机散布的孤立噪点骗去搭桥。有没有一种方法，既能识别像盘山公路一样弯曲的任意非凸形状簇 (Non-convex)，又能在面对漫天噪点时铁面无私地将它们无视？<br>
答案是 <strong>Density-Based (基于密度的聚类)</strong>。该流派认为，簇的本质是空间中被一大片“荒漠区域”隔开的一块块“高密度人口聚集区”。只要人口密度一直连绵不断，无论这片居住区形状多么奇特怪异，它都算作同一个簇群。</p>
<h2>2. DBSCAN 的内功心法与核心实体</h2>
<p>DBSCAN 依赖两个至关重要的超参数：<br>
1. <strong>Eps ($\\epsilon$)</strong>：探测雷达的辐射范围（邻域半径）。<br>
2. <strong>MinPts</strong>：想要成立一个“据点”，在这片邻域内必须包含的最低人口指标（最小样本数，包含自己）。</p>
<h3>数据实体的三六九等判定：</h3>
<ul>
    <li><strong>Core points (核心点)</strong>：王者级别。如果样本拉开 Eps 雷达扫描，发现圈内的人数 $\\ge MinPts$，它立刻晋升为核心点，成为繁华市区的基石。</li>
    <li><strong>Border points (边界点)</strong>：边缘角色。它拉开雷达发现圈内人数惨淡，根本不够 MinPts 指标，无法自立门户。但是，极其幸运的是，它 <strong>刚好处于某一个核心点的雷达辐射圈内</strong>。因此，它被核心点强行收编，成为该聚类簇的边界护城河。</li>
    <li><strong>Noise (噪声点)</strong>：流放者。自己雷达圈内人数极少，同时也悲惨地没有被任何核心点的势力范围所覆盖。DBSCAN 将其无情打上 Noise 标签丢弃。这就是它 <strong>天生拥有强大抗噪鲁棒性</strong> 的奥秘。</li>
</ul>
<h3>组建帝国的规则：</h3>
<p>通过 <strong>密度直达 (Density-reachable)</strong> 关系，所有互相进入彼此辐射圈的核心点，如同烽火台一样互相传递信号，连缀成一个庞大的连通网络，被称为 <strong>密度相连 (Density-connectedness)</strong>。这一整片错综复杂的核心点网络，再加上它们各自收编的那些边界点，构成了一个形态各异、无法预测的巨大单独簇群。</p>
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
    <li><strong>Classes to clusters evaluation (WEKA 的对标黑魔法)</strong>：<br>
    面对算法吐出的一堆名为 Cluster 0, Cluster 1, Cluster 2 的蒙面群落，由于不知道它们究竟对应原先的 猫、狗 还是 猪 类。WEKA 会在后台启动贪婪的 <strong>排列组合匹配</strong>，强行寻找一条能让算法的簇标签与真实的类别标签重合度最高、匹配样本人数最为庞大的映射对位链路。<br>
    一旦完成对位映射，“聚类错配”就会被无缝转化为类似于监督学习那样的 <strong>分类错误率 (Classification error rate)</strong>，从而直接宣告该无监督算法在此数据集上的生杀大权。</li>
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
<h2>3. BUC 算法 (Bottom-Up Construction)：基于单调性的降维灭杀</h2>
<p>这是一个名字具有极大欺骗性的神级算法。虽然叫 Bottom-Up，但在递归树与分割流向上，它是 <strong>从 Lattice 顶端唯一的、包含全宇宙数据的 Apex 节点 (0-D，全局一统) 向下启动</strong>，并逐步通过切入一层又一层的新维度变量来进行 <strong>切块细分 (Partition)</strong> 的。</p>
<p><strong>BUC 极其残暴的剪枝 (Prune) 灵魂之源</strong>：<br>
BUC 的运算极度依赖和 Apriori 算法一脉相承的 <strong>单调性原理 (Monotonicity)</strong>。<br>
思考一下切分的现实意义：一个全集总节点的 <code>Count</code> 数量必然大于等于它细分后的任何一个特定子集分支的数量（广东省的人数绝对大于等于深圳市的人数）。<br>
因此在自顶向下分裂的递归中，一旦探测到：<br>
<strong>当前这个维度的中间节点的 Count / Sum 值已经微弱到跌破了冰山阈值 (Iceberg Threshold)，那么不用再挣扎细看了，它下面再细分产生的所有子子孙孙后代网格节点，其数量绝对不可能反弹回光返照大于这个阈值！</strong><br>
算法立刻手起刀落，在这里执行 <strong>整枝阻断剪除 (Prune)</strong>，整个分支子树的所有剩余层级维度组合将直接被连根跳过、全部免去算力！这使得 BUC 成为处理超高维稀疏数据集的当之无愧的绝对王者。</p>
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
<div class="highlight"><strong>专题说明：</strong>针对可能出现的算法“执行步骤排序题”与 WEKA 算法填空题进行针对性梳理。所有算法均采用 WEKA 官方实现命名。</div>
<h2>1. 决策树建树流 (C4.5 / J48) & REPTree</h2>
<p><strong>J48 核心流排序 (Divide-and-Conquer):</strong></p>
<ol>
    <li>如果当前节点内所有样本 <strong>同属于一个类别</strong>，直接设为叶子节点。</li>
    <li>如果已经 <strong>没有属性可分</strong> 或者样本数少于 <code>minNumObj</code> 参数，按多数表决设为叶节点。</li>
    <li>遍历所有属性，计算 <strong>Gain Ratio</strong> （避免了单纯 Information Gain 的多值偏向）。</li>
    <li>挑选带来 <strong>最大纯度提升 (Best Split)</strong> 的属性作为节点。</li>
    <li>根据取值拆分为互不相交的 <strong>子集 (Subsets)</strong>，<strong>递归 (Recursively)</strong> 调用构建子树。</li>
    <li>整树建完后，激活 <strong>后剪枝 (Post-pruning)</strong>，利用 <code>confidenceFactor</code> 计算折叠误差进行修剪。</li>
</ol>
<p><strong>对比 REPTree:</strong> 核心在于它的剪枝极其简单粗暴——强制切出一块 <strong>Holdout (验证集)</strong>，使用 <strong>Reduced Error Pruning (降低错误率剪枝)</strong>，只要子树在验证集上犯错比变成一个单叶子节点还多，立刻砍掉。</p>
<h2>2. 规则分类器 (JRip / PART / ZeroR)</h2>
<p><strong>ZeroR:</strong> 世界上最废物的算法，不看任何特征，直接输出样本集里最多的类，作为一切算法表现的最低底线。</p>
<p><strong>Sequential Covering (JRip 核心流):</strong></p>
<ol>
    <li>初始化空的规则集。</li>
    <li><strong>[Learn-One-Rule]</strong> 运用 FOIL 增益寻找一条 <strong>最佳</strong> 规则。如果在扩展条件时，模型变得过于累赘导致超过了 <strong>MDL (最小描述长度 64 bits)</strong> 限制，立马停止增加前件。</li>
    <li>将绝佳规则加入规则集。</li>
    <li><strong>[Remove-Covered]</strong> 从训练集中，将被规则正确覆盖的实例 <strong>物理删除 (Separate and Conquer)</strong>。</li>
    <li>只要还有数据，跳回步骤2继续学。</li>
    <li>最后进行 <strong>全局规则优化与替换 (Global Optimization)</strong>。</li>
</ol>
<p><strong>对比 PART:</strong> 它在 <strong>Learn-One-Rule</strong> 的时候，不像 JRip 慢慢拼规则，而是直接在当下的数据上用 C4.5 建一棵局部的“部分决策树”，挑覆盖最广的叶子做成规则，然后立马把树扔了。极其纯粹但极其耗时。</p>
<h2>3. 懒惰学习器 (IBk / KNN)</h2>
<p><strong>算法特点:</strong></p>
<ol>
    <li><strong>训练阶段 (Training):</strong> $O(1)$ 时间复杂度。什么都不做，仅仅把数据写入内存。</li>
    <li><strong>测试阶段 (Testing):</strong> 对新来的实例，计算其与库中每一个点的 <strong>欧氏距离</strong>（必须预先过 <code>Normalize</code> 滤镜）。</li>
    <li>根据 <code>distanceWeighting</code> 参数（如 Inverse Distance）分配权重。</li>
    <li>进行 <strong>多数加权投票</strong>。</li>
</ol>
<h2>4. 频繁项集迭代闭环 (Apriori)</h2>
<p><strong>核心流排序:</strong></p>
<ol>
    <li>扫描数据库，找出现次数 $\\ge Min\\_Sup$ 的单件，得到 <strong>$L_1$</strong>。</li>
    <li><strong>[Join Step]</strong> 让上一轮的 $L_{k-1}$ 自己跟自己配对，拼出下一维度的 <strong>候选集 $C_k$</strong>。</li>
    <li><strong>[Prune Step - 极其关键]</strong> 对于 $C_k$ 中的每个候选项，强行检查它的所有 $(k-1)$ 维子集。如果 <strong>发现任何一个子集</strong> 不在 $L_{k-1}$ 中，利用反单调性定律，立刻将该候选从 $C_k$ <strong>粉碎删除</strong>。</li>
    <li>再扫一次真实数据库，数一数幸存的 $C_k$ 的真实露脸次数，达标的正式晋升为 <strong>$L_k$</strong>。</li>
    <li>循环直到找不到新的项集。</li>
</ol>
<h2>5. 聚类双雄 (SimpleKMeans vs DBSCAN)</h2>
<p><strong>SimpleKMeans (划分流):</strong></p>
<ol>
    <li><strong>随机空投</strong> $K$ 个点作为 Centroids。</li>
    <li><strong>[Assignment]</strong> 遍历所有人，计算距离，收归距离最近的质心麾下。</li>
    <li><strong>[Update]</strong> 重算各个阵营内部的坐标平均值 $\\mu_i$，移动质心到该坐标。</li>
    <li>循环直到 SSE 极小，无人叛变跳槽。</li>
</ol>
<p><strong>DBSCAN (密度流):</strong></p>
<ol>
    <li>标记所有点 <strong>Unvisited</strong>。随机挑一个标记 Visited。</li>
    <li>打开雷达，扫出距离 $\\le Eps$ 的邻居。</li>
    <li>如果邻居 $\\ge MinPts$，该点封神为 <strong>Core Point (核心点)</strong>。建新簇 $C$。</li>
    <li>通过 <strong>Density-reachable (密度可达)</strong> 顺藤摸瓜，遍历它所有邻居，如果邻居也是核心点，就把邻居的邻居全拉进 $C$。拔出萝卜带出泥。</li>
    <li>如果邻居 $< MinPts$，打上 <strong>Noise (噪声)</strong> 标签无情抛弃。</li>
</ol>
<h2>6. BUC 数据立方体裁决流</h2>
<p><strong>核心流排序:</strong></p>
<ol>
    <li>直接从 Lattice 顶端的 <strong>Apex (0-D，全局一统)</strong> 启动算法。</li>
    <li>对当前维度集合进行 <strong>Partition (切分细化)</strong>。</li>
    <li><strong>[Iceberg Pruning 极速阻断]</strong>：核对当前切割出的节点的 Count / Sum 值。一旦发现它 <strong>$\\le Min\\_Support$ (跌破冰山阈值)</strong>。</li>
    <li>运用 <strong>单调性原理</strong>（子集的数量绝对不可能逆势反弹），直接砍断该节点下的所有降维子树，彻底免算！</li>
    <li>对满足条件的节点，继续递归下钻 (Drill-down)。</li>
</ol>
"""

modules_data_js = f"""const modulesData = {{
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

with open('index.html', 'r', encoding='utf-8') as f:
    content = f.read()

start_marker = "const modulesData = {"
end_marker = "function openModule(modId) {"
start_idx = content.find(start_marker)
end_idx = content.find(end_marker)

if start_idx != -1 and end_idx != -1:
    new_content = content[:start_idx] + modules_data_js + "\n\n" + content[end_idx:]
    with open('index.html', 'w', encoding='utf-8') as f:
        f.write(new_content)
    print("Example questions appended to all modules successfully.")
else:
    print("Could not find insertion points.")
