# CS5483 数据挖掘与数据仓库 - 终极期末复习全纪录

本纲要严格按照课程Lectures系列PPT的全部内容逐一整理，旨在100%覆盖讲义中的所有知识点、原理、举例及比较，作为考试前最详尽的复习材料。参考文件范围包括所有的Classification, Cluster Analysis, Data Cube, Data Warehouse, Ensemble Methods, Evaluation, Frequent Pattern, Overview等。

---

## 1. 概述与数据预处理 (Overview & Data Preparation)
* **KDD (Knowledge Discovery from Databases)**：数据挖掘是KDD的核心步骤。
* **KDD步骤**：Preparation (设定目标) $\rightarrow$ Pre-processing (数据清洗、选择、降维、转换) $\rightarrow$ Data mining (应用算法提取模式) $\rightarrow$ Interpretation (解释评估) $\rightarrow$ Deployment (部署)。
* **数据类型 (Attribute Types)**：Nominal/Categorical (标称/类别)，Ordinal (有序)，Numeric/Quantitative (数值/定量)。
* **数据可视化与统计**：散点图、直方图、箱线图、QQ图。统计量包括均值、中位数、众数、方差、四分位数等。
* **距离度量**：Euclidean (欧氏), Manhattan (曼哈顿), Minkowski, Jaccard系数, Cosine余弦相似度等。
* **数据预处理**：处理缺失值、冗余属性（有时有帮助）、剔除无关属性。

---

## 2. 分类模型性能评估 (Classification Evaluation)
* **混淆矩阵 (Confusion Matrix - Binary)**
  * **TP (True Positive)**: 实际正例被预测为正例。
  * **TN (True Negative)**: 实际负例被预测为负例。
  * **FP (False Positive)**: 实际负例被预测为正例 (误报 / False alarm / Type I error)。
  * **FN (False Negative)**: 实际正例被预测为负例 (漏报 / Missed detection / Type II error)。
* **各项核心指标**
  * **Accuracy (准确率)**: $(TP + TN) / (TP + TN + FP + FN)$
  * **Precision (精确率)**: $TP / (TP + FP)$。政府/严格把控方更看重。
  * **Recall (召回率 / TPR)**: $TP / (TP + FN)$。医学诊断中患者更看重，防止漏诊。
  * **NPV (Negative Predictive Value)**: $TN / (TN + FN)$。
  * **Specificity (特异性 / TNR)**: $TN / (TN + FP)$。
* **类别不平衡问题 (Class Imbalance Problem)**: 准确率失效。
  * **F-score**: $2 \times (Precision \times Recall) / (Precision + Recall)$。
  * **F-beta score**: 调整Precision和Recall的权重比例。
* **曲线面积分析 (AUC - Area Under Curve)**
  * 均通过 **Threshold-moving**（改变判定阈值）绘制。
  * **ROC (Receiver Operation Characteristics)**: 纵轴 TPR (Recall) vs 横轴 FPR ($1 - Specificity$)。
  * **PRC (Precision Recall Curve)**: 纵轴 Precision vs 横轴 Recall。在类别极度不平衡时，PRC 比 ROC 更能反映模型好坏。

---

## 3. 分类：决策树 (Classification: Decision Tree)
* **概念**：节点(Node)，根(Root)，叶(Leaves)。预测时沿树从根到叶追踪。
* **分裂算法**：基于“分而治之”的贪心算法(Greedy algorithm)。
* **寻找最佳划分 (Splitting Attribute)**
  * **Gini Impurity (基尼不纯度)**: 衡量随机分类被错分的概率。CART采用。
  * **Shannon's Entropy (信息熵)**: $H(p) = -\sum p_i \log(p_i)$。信息增益 (Information Gain) 偏向于选择分支（Outcomes）极多的属性（如ID），从而导致每次比较减少的杂质极多，但节点泛化差。
  * **补救措施**：使用 Split Information 归一化，即 Gain Ratio (C4.5采用)。或者规定即使是名义(Nominal)变量也只能进行二元划分(Binary split)。
* **连续特征处理**：取连续中点(mid-points)作为分离点。
* **防止过拟合 (Avoiding Overfitting)**
  * **Pre-pruning (预剪枝)**: 在树构建时限制其大小。例如 C4.5 设定最少实例数或信心因子。
  * **Post-pruning (后剪枝)**: 建树后缩减。例如 CART 的 cost-complexity pruning (成本复杂度剪枝)。

---

## 4. 分类：K-近邻学习 (Classification: Nearest Neighbor)
* **思想**：Learning from neighbors, 懒惰学习(Lazy Learners)。
* **决策边界 (Decision Boundaries)**：对于 1-NN 分类器，其决策边界由 **Voronoi 图 (Voronoi diagram)** 决定。
* **标准化 (Normalization)**：不同特征单位（如身高cm vs m）会严重影响距离计算！必须进行 Min-max normalization 或 Standard normalization (Z-score)。
* **距离度量**：选择合适的相似度测量极为关键。
* **优缺点**：
  * *优点*：可以用较少的示例学习复杂的决策边界。
  * *缺点*：如果 $K$ 过小（如 $K=1$），离群点(outlier)会导致严重过拟合；如果 $K$ 过大，会导致欠拟合(underfit)。

---

## 5. 分类：基于规则分类 (Classification: Rule-Based)
* **核心优势**：知识表示具备极强的**模块化 (Modular)**与**可解释性 (Interpretable)**。
* **规则生成方法**：
  1. **从决策树提取**。
  2. **直接生成 (Sequential Covering / 顺序覆盖)**: 采用“分离并征服 (Separate-and-conquer)”。学习一条好规则 $\rightarrow$ 移除被覆盖实例 $\rightarrow$ 循环。
* **具体算法**：
  * **PART (Partial Tree)**：建立一棵局部树，提取能最大化覆盖率(Coverage)的规则。
  * **RIPPER**: 使用FOIL (First Order Inductive Learner) Gain。如何防止过拟合？使用 MDL (最小描述长度)，如果新规则增加超过 64 bits 则停止添加，然后进行整体规则优化。
* **规则排序**：通过基于类的排序 (Class-based ordering) 来解决多条规则发生冲突时的判定问题。

---

## 6. 分类：集成方法 (Classification: Ensemble Methods)
* **概念**：“三个臭皮匠，顶个诸葛亮”。融合多个弱分类器成为强分类器。
* **核心目标**：增加基分类器的**多样性 (Diverse)**以减少偏差 (Reduce Bias)，同时通过组合策略减少方差 (Reduce Variance)。
* **主要架构**：
  * **Bagging (Bootstrap Aggregation)**：
    * 基分类器通过放回抽样(Bootstrap)产生的数据集训练。
    * 合并规则：**多数投票 (Majority voting)** 或 **概率平均 (Average of probabilities)**。
    * 经典变体：**Random Forest (随机森林)**。不仅样本随机，分裂时的特征也随机。
  * **Boosting**：顺序训练模型，重点关注先前错分的样本。例如 **AdaBoost**，使用加权多数投票 (Weighted majority voting)。
  * **Stacking**：训练一个 Meta Classifier (元分类器) 来组合决策。

---

## 7. 模型评估的泛化与过拟合 (Evaluation & The Problem of Overfitting)
* **核心挑战**：分类器在训练数据(Fitted data)上表现优异，但在新数据(Unseen data)上表现糟糕。
* **理论基础**：真实分布未知，只能进行 **Empirical Risk Minimization (经验风险最小化 - ERM)**。使用样本均值估计期望。
* **评估方法**：
  * **Holdout method (保留法)**：直接切分训练与测试集。可能由于特定的切分（不同的Split）导致性能波动。
  * **Stratification (分层)**：按类别的原始比例采样，保证训练集/测试集中类别比例一致。
  * **Cross-Validation (交叉验证)**：多次Holdout取平均。
  * **63.2 Bootstrap**：约63.2%的独特样本被抽入训练集，用来更好地评估。

---

## 8. 频繁模式分析：Apriori 算法 (Frequent Pattern Analysis: Apriori)
* **场景**：市场篮子分析 (Market basket analysis) - 将常一起购买的商品打包，增加库存等。
* **Apriori 性质 (Apriori Property)**：一个频繁项集的所有非空子集也必然是频繁的！如果一个子集不频繁，其超集绝不频繁。
* **核心步骤**：
  1. 扫描计算 1-itemsets。
  2. **Join (连接)**: 使用 $L_{k-1}$ 生成候选 $C_k$。
  3. **Prune (剪枝)**: 根据Apriori性质剪除不满足条件的候选项。
  4. 计算支持度，生成 $L_k$。
* **存储优化**：仅存储 **Maximal frequent itemsets (极大频繁项集)**。极大频繁项集的所有非空子集均为频繁，且它自己不是任何频繁项集的子集。也可存储 Closed itemsets。
* **关联规则生成**：基于支持度和置信度生成规则。**局限性**：高置信度不一定代表强相关，因为可能右侧项本身的出现概率就极高，单纯的支持度-置信度框架存在误导性。

---

## 9. 聚类分析：基于划分与层次 (Cluster Analysis: Partitioning & Hierarchical)
* **Partitioning (划分方法 - K-Means)**
  * 根据距离进行迭代。目标是最小化平方误差。
  * **限制**：极易陷入局部最优；仅能发现凸形(Convex)/球形簇；当簇中心重合或簇形状不规则时完全失效；对离群点敏感。
* **Hierarchical (层次方法)**
  * **Single-linkage (单链/最短距离)**：极易导致 **链式现象 (Chaining phenomenon)**，对异常值和噪声高度敏感。该方法生成的连线等价于 最小生成树 (MST)。
  * **Complete-linkage (全链/最长距离)**：产生更紧凑的簇，但也可能导致簇重叠。计算复杂度高。
  * **AGNES (Agglomerative)**：凝聚式，自底向上合并。
  * **Ward's method**：最小化合并后的方差。
  * 特点：合并是不可逆的 (Irreversible)。

---

## 10. 聚类分析：基于密度 (Cluster Analysis: Density-Based)
* **DBSCAN (Density-Based Spatial Clustering of Applications with Noise)**
  * **核心概念**：Core points (核心点), Border points (边界点), Noise (噪声)。
  * 判断基准：Eps半径内拥有 $\ge$ MinPts 个邻居。基于 Density-reachability (密度可达) 和 Density-connectedness (密度相连)。
  * **优势**：能发现任意形状(非凸)簇，自然识别并过滤噪声，无需预先指定K。
  * **限制**：当面对多种不同密度的簇时表现极差。
* **OPTICS**
  * 为克服DBSCAN对参数高度敏感以及难以处理多密度的问题。
  * 提出 Core-distance (核心距离) 和 Reachability-distance (可达距离)。
  * 输出 **Reachability plot (可达性图)**，其中低谷 (Valley) 代表高密度的簇，完美解决了不同密度的识别问题。

---

## 11. 聚类分析：评估指标 (Cluster Analysis: Evaluation)
* **Tendency (聚类趋势)**：数据本身是否存在簇结构？
* **Intrinsic measures (内在质量指标 / 无Ground Truth)**
  * **Elbow method (肘部法则)**：根据误差平方和的拐点寻找最佳K值，局限在于拐点常常不明显。
  * **Silhouette coefficient (轮廓系数)**：结合了簇内紧密度 $a$ 和簇间分离度 $b$。公式：$(b-a)/\max(a,b)$。
* **Extrinsic measures (外在质量指标 / 有Ground Truth)**
  * **B-Cubed precision and recall**。
  * **Classes to clusters evaluation (如WEKA中)**：将类标签与簇标签最优匹配，计算最小“分类”错误率。

---

## 12. 数据仓库：维度建模与OLAP (Data Warehouse: Data Cube & OLAP)
* **目的**：通过多维视角解决关系型数据库中查询极慢的聚合问题。如寻找销量最高的店和时间。
* **维度建模 (Dimension modeling)**：使用维度 (Time, Location, Item) 作为索引，事实 (Units sold) 作为数据存放在多维数组 (Cuboids) 中。
* **Lattice Structure (晶格/格结构)**：从顶层的所有数据(All)到最底层的具体细节构成的网络。
* **OLAP (联机分析处理) 核心操作**：
  * **Roll-up (上卷)**：减少维度，或沿着概念层次 (Concept hierarchies) 向上攀升。
  * **Drill-down (下钻)**：Roll-up的反向操作。增加细节。
  * **Slice (切片)**：在1个维度上固定选择。
  * **Dice (切块)**：在多个维度或多值上进行切片选择。
  * （OLAP 与 OLTP 区别：OLTP 面向日常事务插入/更新/删除）。

---

## 13. 数据立方体计算 (Data Cube Computation)
* **聚集函数类型**：
  * **Distributive (分配型)**: Count, Sum, Min, Max。父节点可由子节点数据直接聚合而来。
  * **Algebraic (代数型)**: Average。需要多个分配型数据计算。
* **Iceberg Cube (冰山立方体)**：
  * 为避免灾难性的“维度诅咒(Curse of dimensionality)”，只计算满足最小支持度(Count阈值)的单元。
* **BUC 算法 (Bottom-Up Construction)**：
  * 为什么叫 Bottom-Up？因为它从Lattice最底端的Apex（代表聚合了所有数据的单一节点，0-D cuboid）开始，向包含所有维度的Base Cuboid计算。
  * **递归执行**：利用类似Apriori的单调性，一旦某节点的Count小于冰山条件，其下降分区的所有子节点必定不满足阈值，从而进行**整支剪枝 (Prune)**。极度适合稀疏数据。
* **CCIC (Closed Cube and Iceberg Cube)**：
  * 使用 Ancestor-descendant (祖先后代) 关系，记录**闭合立方体 (Closed Cube)**以进一步压缩空间（不记录拥有相同聚合结果的分支节点）。使用 Hass diagram 表示。
