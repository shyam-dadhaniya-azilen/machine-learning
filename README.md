# Machine Learning Algorithms — Comprehensive Guide

## Table of Contents

### Part 1 — Supervised Learning: Regression

| #   | Algorithm                                                      | Key Technique                   |
| --- | -------------------------------------------------------------- | ------------------------------- |
| 1   | [Linear Regression](#1-linear-regression)                      | MSE minimization, least squares |
| 2   | [Polynomial Regression](#2-polynomial-regression)              | Feature degree expansion        |
| 3   | [Ridge Regression (L2)](#3-ridge-regression-l2-regularization) | L2 coefficient penalty          |
| 4   | [Lasso Regression (L1)](#4-lasso-regression-l1-regularization) | L1 penalty, feature selection   |
| 5   | [Elastic Net](#5-elastic-net)                                  | Combined L1 + L2 penalty        |

### Part 2 — Supervised Learning: Classification

| #   | Algorithm                                                       | Key Technique                            |
| --- | --------------------------------------------------------------- | ---------------------------------------- |
| 6   | [Logistic Regression](#6-logistic-regression)                   | Sigmoid function, log-loss               |
| 7   | [Support Vector Machines (SVM)](#7-support-vector-machines-svm) | Max-margin hyperplane, kernel trick      |
| 8   | [Naive Bayes](#8-naive-bayes)                                   | Bayes' theorem, conditional independence |
| 9   | [k-Nearest Neighbors (k-NN)](#9-k-nearest-neighbors-k-nn)       | Euclidean distance, majority vote        |
| 10  | [Decision Trees](#10-decision-trees)                            | Gini impurity, recursive splitting       |

### Part 3 — Ensemble Methods

| #   | Algorithm                                                              | Key Technique                         |
| --- | ---------------------------------------------------------------------- | ------------------------------------- |
| 11  | [Random Forest](#11-random-forest-bagging)                             | Bagging, parallel trees               |
| 12  | [Gradient Boosting Machines (GBM)](#12-gradient-boosting-machines-gbm) | Sequential residual fitting           |
| 13  | [XGBoost](#13-xgboost-extreme-gradient-boosting)                       | Regularized boosting, parallel splits |
| 14  | [AdaBoost](#14-adaboost-adaptive-boosting)                             | Adaptive sample reweighting           |

### Part 4 — Unsupervised Learning: Clustering

| #   | Algorithm                                              | Key Technique                                |
| --- | ------------------------------------------------------ | -------------------------------------------- |
| 15  | [k-Means Clustering](#15-k-means-clustering)           | WCSS minimization, centroid iteration        |
| 16  | [Hierarchical Clustering](#16-hierarchical-clustering) | Dendrogram, Ward's linkage                   |
| 17  | [DBSCAN](#17-dbscan)                                   | Density-based neighborhoods, noise detection |

### Part 5 — Unsupervised Learning: Dimensionality Reduction

| #   | Algorithm                                                                  | Key Technique                             |
| --- | -------------------------------------------------------------------------- | ----------------------------------------- |
| 18  | [Principal Component Analysis (PCA)](#18-principal-component-analysis-pca) | Eigendecomposition, variance maximization |
| 19  | [t-SNE](#19-t-distributed-stochastic-neighbor-embedding-t-sne)             | Student's t-kernel, KL divergence         |

### Part 6 — Reinforcement Learning

| #   | Algorithm                    | Key Technique                         |
| --- | ---------------------------- | ------------------------------------- |
| 20  | [Q-Learning](#20-q-learning) | Bellman equation, temporal difference |

---

### Decision Map: Which Algorithm to Use?

```
Start: What is your problem type?
│
├── 🏷️  LABELED DATA (Supervised Learning)
│   │
│   ├── 📈  Output is a NUMBER (Regression)
│   │   ├── Linear relationship?              → Linear Regression
│   │   ├── Curved / non-linear relationship? → Polynomial Regression
│   │   ├── Many correlated features?         → Ridge Regression
│   │   ├── Need automatic feature selection? → Lasso / Elastic Net
│   │   └── Need best possible accuracy?      → XGBoost / GBM
│   │
│   └── 🏷️  Output is a CATEGORY (Classification)
│       ├── Need probability output?           → Logistic Regression
│       ├── High-dim data, small dataset?      → SVM (with RBF kernel)
│       ├── Text data or fast baseline?        → Naive Bayes
│       ├── Need full interpretability?        → Decision Tree
│       ├── No training needed, simple?        → k-NN
│       └── Best general accuracy?             → Random Forest / XGBoost
│
└── 🔍  NO LABELS (Unsupervised Learning)
    │
    ├── 🔵  FIND GROUPS (Clustering)
    │   ├── Know number of groups K?            → k-Means
    │   ├── Want a full cluster hierarchy?      → Hierarchical Clustering
    │   └── Irregular shapes + detect outliers? → DBSCAN
    │
    ├── 📉  REDUCE DIMENSIONS
    │   ├── For downstream ML models?           → PCA
    │   └── For visualization only (2D/3D)?     → t-SNE
    │
    └── 🎮  AGENT IN ENVIRONMENT
        └── Discrete states + actions?          → Q-Learning
```

### 60-Second Daily Review Template

```
Algorithm           | Hook       | Formula Core       | Use When
--------------------+------------+--------------------+---------------------------
Linear Regression   | LINE       | Y = β₀ + β₁X       | Continuous output, linear
Polynomial Reg.     | CURVE      | Y = β₀ + β₁X + βₙXⁿ| Non-linear curve needed
Ridge               | SHRINK     | MSE + αΣβ²         | Correlated features
Lasso               | DELETE     | MSE + αΣ|β|        | Feature selection needed
Elastic Net         | BLEND      | MSE + L1 + L2      | Both above
Logistic Reg.       | S-CURVE    | P = 1/(1+e^-z)     | Binary classification
SVM                 | CORRIDOR   | max 2/||w||        | High-dim, small data
Naive Bayes         | MULTIPLY   | P(C)·ΠP(Xᵢ|C)     | Text classification
k-NN                | NEIGHBORS  | d = √Σ(p-q)²      | No-training classifier
Decision Tree       | QUESTIONS  | Gini = 1-Σpᵢ²     | Interpretability needed
Random Forest       | VOTE       | ŷ = mode{fᵦ(x)}   | Reduce overfitting
GBM                 | CORRECT    | Fₘ = Fₘ₋₁ + γhₘ  | High accuracy, tabular
XGBoost             | TURBO      | L + Ω(f)          | Best accuracy + speed
AdaBoost            | FOCUS      | wᵢ·exp(−αₜyᵢhₜ)  | Weak classifiers combined
k-Means             | CENTROIDS  | min WCSS           | Known K, spherical clusters
Hierarchical        | MERGE      | Ward's Δ(A,B)      | Unknown K, hierarchy needed
DBSCAN              | DENSITY    | |N_ε(p)| ≥ MinPts  | Arbitrary shapes, outliers
PCA                 | ROTATE     | Σv = λv            | Dimensionality reduction
t-SNE               | PULL/PUSH  | q_ij (t-dist)      | 2D/3D visualization only
Q-Learning          | REWARD     | Q ← Q+α[R+γmaxQ']  | Agent, discrete env
```

---

## Part 1: Supervised Learning — Regression

> Algorithms that predict a **continuous numeric output** from labeled training data.

### 1. Linear Regression

Assumes a linear relationship between input variables ($X$) and a continuous target variable ($Y$).

- **Formula**:
  $$Y = \beta_0 + \beta_1X_1 + \beta_2X_2 + \dots + \beta_nX_n + \epsilon$$
  Weights are optimized by minimizing **Mean Squared Error (MSE)**:
  $$MSE = \frac{1}{m} \sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$
- **Components**:
  - $Y$: predicted output (dependent variable)
  - $\beta_0$: bias/intercept term
  - $\beta_1, \dots, \beta_n$: learned coefficients for each input feature
  - $X_1, \dots, X_n$: input features (independent variables)
  - $\epsilon$: irreducible error (noise) term
  - $m$: number of training samples
  - $y_i$: actual (true) value for sample $i$
  - $\hat{y}_i$: model's predicted value for sample $i$
- **Example**: Predicting a software engineer's salary based on years of experience.
  $$\text{Salary} = 30,000 + 5,000 \times (\text{Years})$$
  For 4 years of experience: $30,000 + 5,000(4) = \$50,000$.
- **Key Characteristics**:
  - Simple and highly interpretable — coefficients show direct feature impact
  - Assumes a strictly linear relationship between features and target
  - Sensitive to outliers (MSE penalizes large errors quadratically)
  - Requires feature scaling for fair coefficient comparison
  - No hyperparameters; computationally very efficient (closed-form solution)
  - Can underfit complex non-linear data

---

### 2. Polynomial Regression

Fits a non-linear relationship by transforming features into higher-degree polynomials.

- **Formula**:
  $$Y = \beta_0 + \beta_1X + \beta_2X^2 + \dots + \beta_dX^d + \epsilon$$
- **Components**:
  - $Y$: predicted output
  - $\beta_0, \dots, \beta_d$: polynomial coefficients learned during training
  - $X, X^2, \dots, X^d$: polynomial expansions of the input feature up to degree $d$
  - $d$: degree of the polynomial (hyperparameter controlling curve complexity)
  - $\epsilon$: error/noise term
- **Example**: Modeling epidemic spread over time, where infection rates accelerate non-linearly before peaking.
- **Key Characteristics**:
  - Captures non-linear patterns without changing the linear regression framework
  - Higher degree $d$ → more flexible curve but higher risk of overfitting
  - Degree $d$ is a hyperparameter that must be tuned (e.g., via cross-validation)
  - Interpretable coefficients but increasingly complex at high degrees
  - Sensitive to outliers at the extremes of the feature range

---

### 3. Ridge Regression (L2 Regularization)

Prevents overfitting and handles multicollinearity by adding a penalty proportional to the _square_ of the magnitude of coefficients.

- **Formula**:
  $$\text{Loss} = \text{MSE} + \alpha \sum_{j=1}^{n} \beta_j^2$$
- **Components**:
  - $\text{MSE}$: standard Mean Squared Error of predictions
  - $\alpha$: regularization strength (hyperparameter; higher = stronger shrinkage)
  - $\beta_j$: coefficient of the $j$-th feature
  - $\sum \beta_j^2$: L2 penalty — sum of squared coefficients, shrinks them toward zero
- **Example**: Predicting house prices using 100 highly correlated real estate features; shrinks coefficients to prevent wild model sensitivity.
- **Key Characteristics**:
  - Shrinks coefficients toward zero but **never exactly to zero** (retains all features)
  - Excellent for handling multicollinearity (correlated features)
  - Closed-form analytical solution exists (computationally efficient)
  - Does not perform feature selection — use Lasso when sparsity is needed
  - Single hyperparameter $\alpha$ controls regularization strength

---

### 4. Lasso Regression (L1 Regularization)

Prevents overfitting by adding a penalty proportional to the _absolute value_ of coefficients. This drives less important weights to exactly zero, performing automatic feature selection.

- **Formula**:
  $$\text{Loss} = \text{MSE} + \alpha \sum_{j=1}^{n} |\beta_j|$$
- **Components**:
  - $\text{MSE}$: standard Mean Squared Error of predictions
  - $\alpha$: regularization strength (higher = more aggressive feature zeroing)
  - $|\beta_j|$: absolute value of the $j$-th coefficient (L1 penalty)
  - $\sum |\beta_j|$: L1 penalty — drives irrelevant coefficients to exactly zero
- **Example**: Predicting patient health outcomes using thousands of genetic markers; zeroes out irrelevant genes to isolate the key indicators.
- **Key Characteristics**:
  - Performs **automatic feature selection** by driving irrelevant coefficients to exactly zero
  - Produces sparse, interpretable models ideal for high-dimensional data
  - No closed-form solution; solved via coordinate descent
  - Can be unstable when features are highly correlated (randomly selects one from a group)
  - Single hyperparameter $\alpha$; larger values remove more features

---

### 5. Elastic Net

Combines both L1 (Lasso) and L2 (Ridge) penalties to balance the benefits of feature selection and structural stability.

- **Formula**:
  $$\text{Loss} = \text{MSE} + \alpha \cdot \rho \sum_{j=1}^{n} |\beta_j| + \frac{\alpha \cdot (1 - \rho)}{2} \sum_{j=1}^{n} \beta_j^2$$
  _(where $\rho$ controls the balance between L1 and L2 features)_
- **Components**:
  - $\text{MSE}$: standard Mean Squared Error of predictions
  - $\alpha$: overall regularization strength
  - $\rho$: mixing ratio — $\rho=1$ gives pure Lasso (L1); $\rho=0$ gives pure Ridge (L2)
  - $\alpha \cdot \rho \sum |\beta_j|$: L1 penalty component (feature selection)
  - $\frac{\alpha(1-\rho)}{2} \sum \beta_j^2$: L2 penalty component (coefficient shrinkage)
- **Example**: Analyzing marketing multi-channel spend data where features are highly correlated and grouped.
- **Key Characteristics**:
  - Best of both worlds: feature selection (L1) + coefficient stability (L2)
  - Handles correlated features more gracefully than pure Lasso
  - Two hyperparameters ($\alpha$, $\rho$) require more careful tuning
  - Preferred when feature groups exist and group-level selection is desired
  - Falls back to Ridge ($\rho=0$) or Lasso ($\rho=1$) at extremes

---

## Part 2: Supervised Learning — Classification

> Algorithms that predict a **discrete class label** from labeled training data.

### 6. Logistic Regression

Maps continuous inputs to output probabilities between $0$ and $1$ using a sigmoid function for classification tasks.

- **Formula**:
  $$P(Y=1|X) = \frac{1}{1 + e^{-z}} \quad \text{where} \quad z = \beta_0 + \beta_1X_1 + \dots + \beta_nX_n$$
  Optimized via **Binary Cross-Entropy (Log Loss)**:
  $$\text{Loss} = -\frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$
- **Components**:
  - $P(Y=1|X)$: predicted probability that sample $X$ belongs to class 1
  - $z$: linear combination of inputs (the logit)
  - $e$: Euler's number (~2.718); $e^{-z}$ maps any real number to $(0,1)$
  - $\beta_0$: bias term; $\beta_1, \dots, \beta_n$: learned feature weights
  - $m$: number of training samples
  - $y_i$: true binary label (0 or 1) for sample $i$
  - $\hat{y}_i$: predicted probability for sample $i$
- **Example**: Predicting email classification. If an email yields $z = 2.5$:
  $$P(\text{Spam}) = \frac{1}{1 + e^{-2.5}} \approx 0.923 \implies \mathbf{Spam} \ \text{(since } \ge 0.5\text{)}$$
- **Key Characteristics**:
  - Outputs calibrated probabilities, not just class labels
  - Assumes a **linear decision boundary** in feature space
  - Highly interpretable — coefficient sign and magnitude indicate feature direction/importance
  - Fast to train; scales well to large datasets
  - Sensitive to class imbalance and requires feature scaling
  - Extend to multi-class via One-vs-Rest (OvR) or Softmax

---

### 7. Support Vector Machines (SVM)

Finds the optimal hyperplane that separates classes with the maximum possible margin.

- **Formula**:
  The decision boundary satisfies $\mathbf{w} \cdot \mathbf{x} + b = 0$. Optimization maximizes the margin $\frac{2}{||\mathbf{w}||}$ subject to:
  $$y_i(\mathbf{w} \cdot \mathbf{x}_i + b) \ge 1 \quad \forall i$$
  _Non-linear variations use Kernel functions like the RBF Kernel:_ $K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma ||\mathbf{x} - \mathbf{x}'||^2)$
- **Components**:
  - $\mathbf{w}$: weight vector — normal (perpendicular) to the decision hyperplane
  - $b$: bias scalar that shifts the hyperplane
  - $\frac{2}{||\mathbf{w}||}$: margin width between the two support-vector boundaries
  - $y_i \in \{-1, +1\}$: true class label for training sample $i$
  - $\mathbf{x}_i$: feature vector for sample $i$
  - $\gamma$: RBF kernel bandwidth — controls how far influence of each training point reaches
  - $||\mathbf{x} - \mathbf{x}'||^2$: squared Euclidean distance between two input vectors
- **Example**: Classifying medical images as benign or malignant based on boundary margin lines.
- **Key Characteristics**:
  - Maximizes the **margin** between classes, making it robust to outliers near boundaries
  - The kernel trick enables non-linear classification without explicitly computing high-dim features
  - Effective in high-dimensional spaces (e.g., text, images)
  - Training is slow for large datasets ($O(n^2)$ to $O(n^3)$); not ideal for $n > 100{,}000$
  - Requires careful feature scaling (mandatory) and hyperparameter tuning ($C$, $\gamma$)
  - Memory-intensive: stores only support vectors after training

---

### 8. Naive Bayes

Uses Bayes' Theorem assuming that all input features are completely independent given the target class label.

- **Formula**:
  $$\hat{y} = \arg\max_{k} P(C_k) \prod_{i=1}^{n} P(X_i | C_k)$$
- **Components**:
  - $\hat{y}$: predicted class label
  - $\arg\max_k$: selects the class $k$ with the highest posterior score
  - $P(C_k)$: prior probability of class $k$ (class frequency in training data)
  - $P(X_i | C_k)$: likelihood of observing feature $X_i$ given class $k$
  - $\prod_{i=1}^{n}$: product over all features (the naive independence assumption)
- **Example**: Classifying text sentiment. If a message contains the word "great":
  $$\text{Score(Positive)} = P(\text{Pos}) \times P(\text{"great"|Pos}) = 0.5 \times 0.3 = \mathbf{0.15}$$
  $$\text{Score(Negative)} = P(\text{Neg}) \times P(\text{"great"|Neg}) = 0.5 \times 0.01 = 0.005$$
  Result: **Positive** sentiment classification.
- **Key Characteristics**:
  - Extremely fast training — just stores the data (no model parameters to learn)
  - Works surprisingly well despite the strong independence assumption
  - Optimal for text classification (Multinomial), Gaussian continuous data (GaussianNB)
  - Performs poorly when features are highly correlated or dependent
  - Requires smoothing (e.g., Laplace smoothing) to handle zero-probability features

---

### 9. k-Nearest Neighbors (k-NN)

Classifies a test instance based on a majority vote of its geographically closest training points.

- **Formula** (Euclidean Distance):
  $$d(p, q) = \sqrt{\sum_{i=1}^{n} (p_i - q_i)^2}$$
- **Components**:
  - $d(p, q)$: straight-line (Euclidean) distance between points $p$ and $q$
  - $p_i, q_i$: the $i$-th feature coordinate of points $p$ and $q$ respectively
  - $n$: total number of features (dimensions)
  - $k$: number of nearest neighbors to consider for the majority vote
- **Example**: Categorizing a new fruit with Sweetness 6, Firmness 7 using $k=3$. The 3 closest data neighbors are: 2 Oranges and 1 Apple. Classification: **Orange**.
- **Key Characteristics**:
  - **Lazy learner** — no training phase; all computation happens at prediction time
  - Non-parametric: makes no assumptions about data distribution
  - Prediction cost is $O(n \cdot d)$ per query — slow for large datasets
  - Very sensitive to feature scaling (use standardization or normalization)
  - Performance degrades in high dimensions (curse of dimensionality)
  - Odd values of $k$ are preferred to avoid ties in binary classification

---

### 10. Decision Trees

Recursively splits datasets into distinct groups using rule-based conditions based on structural purity metrics.

- **Formula** (Gini Impurity):
  $$Gini = 1 - \sum_{i=1}^{C} (p_i)^2$$
  _(where $p_i$ is the probability of an item belonging to class $i$)_
- **Components**:
  - $Gini$: impurity score — 0 means perfectly pure node, ~0.5 means maximally mixed
  - $C$: number of distinct class labels
  - $p_i$: proportion of training samples in the current node that belong to class $i$
  - $\sum (p_i)^2$: sum of squared class probabilities; higher = purer node
- **Example**: Determining loan approval. Split 1: Is Credit Score $> 700$? If Yes $\rightarrow$ Approve. If No $\rightarrow$ Check Income level.
- **Key Characteristics**:
  - Highly interpretable — rules can be visualized and explained to non-technical stakeholders
  - No feature scaling required (splits are threshold-based)
  - Handles both numerical and categorical features natively
  - Prone to **overfitting** on training data (deep trees memorize noise)
  - Greedy splitting: locally optimal splits may not be globally optimal
  - Controlled via `max_depth`, `min_samples_split`, and pruning strategies

---

## Part 3: Ensemble Methods

> Algorithms that **combine multiple models** to produce stronger predictions than any single model.

### 11. Random Forest (Bagging)

Builds many deep independent decision trees in parallel using bootstrapped training subsets and averages their predictions to reduce variance.

- **Formula**:
  $$\hat{y} = \frac{1}{B} \sum_{b=1}^{B} f_b(x) \quad \text{(Regression)} \quad \text{or} \quad \hat{y} = \text{mode}\{f_1(x), \dots, f_B(x)\} \quad \text{(Classification)}$$
- **Components**:
  - $\hat{y}$: final ensemble prediction
  - $B$: total number of decision trees in the forest
  - $f_b(x)$: prediction from the $b$-th individual tree for input $x$
  - $\frac{1}{B} \sum f_b(x)$: average over all trees (used for regression)
  - $\text{mode}\{\cdot\}$: majority class vote across all trees (used for classification)
- **Example**: Predicting credit card fraud by averaging predictions from 500 distinct, randomized decision tree variations.
- **Key Characteristics**:
  - Reduces variance (overfitting) through **bagging** — each tree sees a random data subset
  - Random feature subsampling at each split further decorrelates trees
  - Robust to outliers and missing values
  - Provides built-in **feature importance** scores
  - Trees are trained independently → highly **parallelizable**
  - Less interpretable than a single decision tree

---

### 12. Gradient Boosting Machines (GBM)

Trains weak learner decision trees sequentially. Each new tree fits to the negative gradient of the loss function (the residuals/errors) of the collective model.

- **Formula**:
  $$F_m(x) = F_{m-1}(x) + \gamma_m h_m(x)$$
  _(where $h_m(x)$ is a tree trained on residuals, and $\gamma_m$ is the step size)_
- **Components**:
  - $F_m(x)$: updated ensemble model after adding the $m$-th tree
  - $F_{m-1}(x)$: the existing ensemble model from the previous iteration
  - $h_m(x)$: the new weak learner (tree) fitted to the current residuals/errors
  - $\gamma_m$: learning rate (step size) scaling the contribution of $h_m(x)$; prevents overfitting
- **Example**: House price prediction where Tree 2 fixes the exact dollar pricing mistakes made by Tree 1.
- **Key Characteristics**:
  - Sequentially corrects errors of previous models — reduces both bias and variance
  - Typically achieves **higher accuracy** than Random Forest on structured data
  - Trees are built sequentially → **not parallelizable** across iterations
  - Sensitive to hyperparameters: learning rate ($\gamma_m$), tree depth, number of trees
  - Prone to overfitting if too many trees are added without regularization
  - Supports multiple loss functions (MSE for regression, log-loss for classification)

---

### 13. XGBoost (Extreme Gradient Boosting)

An advanced, highly scaled version of gradient boosting engineered with hardware optimizations and built-in L1/L2 tree regularization.

- **Formula** (Objective Function):
  $$\mathcal{L}^{(t)} = \sum_{i=1}^{m} l(y_i, \hat{y}_i^{(t-1)} + f_t(x_i)) + \Omega(f_t) \quad \text{where} \quad \Omega(f) = \gamma T + \frac{1}{2}\lambda \sum_{j=1}^{T} w_j^2$$
- **Components**:
  - $\mathcal{L}^{(t)}$: total objective loss at boosting iteration $t$
  - $l(y_i, \cdot)$: per-sample loss function (e.g., squared error) for true label $y_i$
  - $\hat{y}_i^{(t-1)}$: cumulative prediction before adding tree $f_t$
  - $f_t(x_i)$: new tree's prediction for sample $i$
  - $\Omega(f_t)$: regularization term penalizing tree complexity
  - $T$: number of leaves in tree $f_t$
  - $w_j$: output score (weight) of leaf $j$
  - $\gamma$: minimum loss reduction required for a split (pruning threshold)
  - $\lambda$: L2 regularization coefficient on leaf weights
- **Example**: High-speed win-prediction engines built for competitive data science tournaments.
- **Key Characteristics**:
  - **Faster** than standard GBM due to approximate split finding and parallel tree construction
  - Built-in L1/L2 regularization via $\gamma$ and $\lambda$ prevents overfitting
  - Handles **missing values natively** without imputation
  - Supports custom loss functions and evaluation metrics
  - Column (feature) sub-sampling reduces memory usage and adds randomness
  - Dominant algorithm in structured/tabular data competitions (Kaggle)

---

### 14. AdaBoost (Adaptive Boosting)

Constructs sequential models (usually shallow "decision stumps") by progressively increasing the error weights of previously misclassified records.

- **Formula** (Sample Weight Update):
  $$w_i^{(t+1)} = w_i^{(t)} \cdot \exp(-\alpha_t y_i h_t(x_i))$$
- **Components**:
  - $w_i^{(t+1)}$: updated sample weight after round $t$
  - $w_i^{(t)}$: current weight of sample $i$ — misclassified samples get higher weights
  - $\alpha_t$: confidence weight of the $t$-th weak learner (higher accuracy → larger $\alpha_t$)
  - $y_i \in \{-1, +1\}$: true class label for sample $i$
  - $h_t(x_i)$: prediction of the $t$-th weak learner on sample $i$
  - $\exp(\cdot)$: exponential function; correct predictions reduce weight, errors increase it
- **Example**: Detecting human faces in imagery by linking basic edge-detection filters that emphasize hard-to-classify samples.
- **Key Characteristics**:
  - Combines many **weak classifiers** into one strong ensemble
  - Works with any base learner (commonly decision stumps — depth-1 trees)
  - Highly **sensitive to noisy data and outliers** (upweights hard/mislabeled samples)
  - Less prone to overfitting than a single deep tree
  - Final prediction is a **weighted majority vote** of all weak classifiers
  - Performance degrades when base classifier accuracy falls below 50%

---

## Part 4: Unsupervised Learning — Clustering

> Algorithms that **discover hidden groupings** in unlabeled data.

### 15. k-Means Clustering

Partitions unlabeled coordinates into $K$ uniform groups by iteratively minimizing point distances to geometric centroids.

- **Formula** (Within-Cluster Sum of Squares - WCSS):
  $$\text{WCSS} = \sum_{j=1}^{K} \sum_{i \in S_j} ||x_i - \mu_j||^2$$
- **Components**:
  - $K$: number of clusters (chosen by the user)
  - $S_j$: set of all data points assigned to cluster $j$
  - $x_i$: feature vector of data point $i$
  - $\mu_j$: centroid (mean position) of cluster $j$
  - $||x_i - \mu_j||^2$: squared Euclidean distance from point $i$ to its cluster centroid
  - $\text{WCSS}$: objective to minimize — lower value = tighter, more compact clusters
- **Example**: Profiling retail customers into $K=3$ groups based on spatial similarity in spending vs. store visit frequency.
- **Key Characteristics**:
  - Simple and computationally efficient — scales well to large datasets
  - Requires specifying $K$ in advance (use the **Elbow Method** on WCSS to find optimal $K$)
  - Sensitive to **initialization** — use k-means++ for smarter centroid seeding
  - Assumes **spherical, equally-sized clusters** (struggles with irregular shapes)
  - Sensitive to outliers (outliers distort centroid positions)
  - Results vary between runs due to random initialization

---

### 16. Hierarchical Clustering

Builds a multi-layered cluster tree structure (Dendrogram) linking samples through either bottom-up (agglomerative) or top-down (divisive) strategies.

- **Formula** (Ward's Linkage Method):
  $$\Delta(A, B) = \frac{n_A n_B}{n_A + n_B} ||\mu_A - \mu_B||^2$$
- **Components**:
  - $\Delta(A, B)$: merge cost (dissimilarity) between clusters $A$ and $B$
  - $n_A, n_B$: number of data points in clusters $A$ and $B$ respectively
  - $\frac{n_A n_B}{n_A + n_B}$: harmonic-style weighting that penalizes merging large clusters
  - $\mu_A, \mu_B$: centroids (mean vectors) of clusters $A$ and $B$
  - $||\mu_A - \mu_B||^2$: squared Euclidean distance between the two centroids
- **Example**: Creating a biological taxonomy tree grouping newly discovered plant samples based on physical feature matrices.
- **Key Characteristics**:
  - No need to specify $K$ in advance — cut the dendrogram at any level to get clusters
  - Produces a **dendrogram** for full visual exploration of cluster hierarchy
  - **Deterministic** — same data always produces the same result
  - Computationally expensive: $O(n^2)$ time and memory — not suitable for very large datasets
  - Sensitive to choice of linkage method (single, complete, average, Ward's)
  - Cannot update incrementally with new data

---

### 17. DBSCAN

Groups items based on dense spatial clusters, leaving sparse isolated points behind as designated outlier noise.

- **Formula**:
  A point $p$ is a Core Point if:
  $$|N_\epsilon(p)| \ge \text{MinPts} \quad \text{where} \quad N_\epsilon(p) = \{q \in D \mid d(p,q) \le \epsilon\}$$
- **Components**:
  - $p$: the query data point being evaluated
  - $\epsilon$: neighborhood radius — the maximum distance to consider two points "neighbors"
  - $N_\epsilon(p)$: set of all points in dataset $D$ within distance $\epsilon$ of $p$
  - $|N_\epsilon(p)|$: count of neighbors within radius $\epsilon$
  - $\text{MinPts}$: minimum neighbor count required to classify $p$ as a core point
  - $D$: the full input dataset
  - Points not reachable from any core point are labeled as **noise/outliers**
- **Example**: Detecting geographic anomalies or mapping out irregularly shaped city congestion zones via GPS logs.
- **Key Characteristics**:
  - Discovers **arbitrarily shaped clusters** (not limited to spherical clusters like k-Means)
  - Automatically identifies and labels **outliers as noise** — no forced assignment
  - No need to pre-specify the number of clusters
  - Sensitive to hyperparameters $\epsilon$ and $\text{MinPts}$ — tuning is non-trivial
  - Struggles in **high-dimensional spaces** (distances become less meaningful)
  - Does not work well when clusters have widely varying densities

---

## Part 5: Unsupervised Learning — Dimensionality Reduction

> Algorithms that **compress high-dimensional data** into fewer dimensions while preserving structure.

### 18. Principal Component Analysis (PCA)

A linear matrix transformation that projects highly dimensional features onto orthogonal vectors to maximize data variance.

- **Formula**:
  Given a covariance matrix $\Sigma$, find eigenvectors $v$ and eigenvalues $\lambda$ satisfying:
  $$\Sigma v = \lambda v$$
- **Components**:
  - $\Sigma$: covariance matrix of the input data (captures feature correlations)
  - $v$: eigenvector — defines a principal component direction in feature space
  - $\lambda$: eigenvalue — magnitude of variance explained along eigenvector $v$
  - Eigenvectors are sorted by descending $\lambda$; top $k$ are kept as principal components
- **Example**: Compressing an image collection from $1000 \times 1000$ pixel arrays down to 50 primary structural component fields.
- **Key Characteristics**:
  - **Unsupervised** — does not use class labels during transformation
  - Components are **orthogonal** (uncorrelated) and ordered by variance explained
  - Only captures **linear** relationships — use Kernel PCA for non-linear structure
  - Requires feature standardization before application
  - Reduces overfitting and speeds up downstream model training
  - Loses interpretability: new axes are combinations of original features

---

### 19. t-Distributed Stochastic Neighbor Embedding (t-SNE)

A non-linear reduction algorithm that maps high-dimensional datasets down to 2D or 3D spaces while maintaining local point affinities.

- **Formula** (Low-dimensional similarity):
  $$q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k} \sum_{l \neq k} (1 + ||y_k - y_l||^2)^{-1}}$$
- **Components**:
  - $q_{ij}$: similarity score between points $i$ and $j$ in the low-dimensional (2D/3D) space
  - $y_i, y_j$: embedded 2D/3D coordinates of data points $i$ and $j$
  - $(1 + ||y_i - y_j||^2)^{-1}$: Student's t-distribution kernel — gives heavier tails to avoid "crowding"
  - Denominator: normalization sum over all other distinct point pairs in the embedding
  - The algorithm minimizes KL divergence between high-dim similarities $p_{ij}$ and $q_{ij}$
- **Example**: Mapping thousands of complex gene expressions onto a standard 2D scatterplot chart to visually detect cellular cluster patterns.
- **Key Characteristics**:
  - Excellent at preserving **local structure** (nearby points stay together in 2D/3D)
  - Uses Student's t-distribution to avoid the **crowding problem** in low-dim space
  - Primarily a **visualization tool** — not suitable for general dimensionality reduction pipelines
  - **Cannot project new data points** (no out-of-sample extension by default)
  - Computationally expensive: $O(n^2)$ — use on samples $\le$ 10,000 for practicality
  - Results change with different `perplexity` and random seed settings

---

## Part 6: Reinforcement Learning

> Algorithms where an **agent learns by interacting** with an environment to maximize cumulative reward.

### 20. Q-Learning

A model-free algorithm where an environment agent learns an optimal action policy by maximizing numerical rewards recorded inside a state-action lookup matrix.

- **Formula** (Bellman Equation Update):
  $$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ R(s, a) + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$
- **Components**:
  - $Q(s, a)$: current quality estimate for taking action $a$ in state $s$
  - $\alpha$: learning rate ($0 < \alpha \le 1$) — controls how fast the Q-value is updated
  - $R(s, a)$: immediate reward received after taking action $a$ from state $s$
  - $\gamma$: discount factor ($0 \le \gamma \le 1$) — weights the importance of future rewards
  - $s'$: the next state the agent transitions to after action $a$
  - $\max_{a'} Q(s', a')$: best possible Q-value achievable from the next state $s'$
  - $[R + \gamma \max Q' - Q]$: temporal difference (TD) error — the correction applied to $Q(s,a)$
- **Key Characteristics**:
  - **Model-free** — learns without any prior knowledge of environment dynamics
  - **Off-policy** — learns the optimal policy regardless of the agent's current exploration behavior
  - Tabular: Q-values stored in a table → limited to **discrete, low-dimensional** state/action spaces
  - Convergence guaranteed under sufficient exploration and decaying learning rate
  - Can be slow to converge in large or continuous state spaces
  - Extended by **Deep Q-Networks (DQN)** which use neural networks to approximate $Q(s,a)$
