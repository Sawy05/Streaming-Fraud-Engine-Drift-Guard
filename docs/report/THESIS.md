# Real-Time Event-Driven Machine Learning Operations Pipeline for Financial Fraud Detection with Point-in-Time Feature Store Integration and Continuous Drift Guard

**Author:** Ahmed Elsawy (`@Sawy05`)  
**Affiliation:** Principal AI Research Scientist & Staff ML Systems Engineer  
**Repository:** [Sawy05/Streaming-Fraud-Engine-Drift-Guard](https://github.com/Sawy05/Streaming-Fraud-Engine-Drift-Guard)  
**Date:** October 2026  
**Document Classification:** Technical Thesis, Architecture Proposal & Empirical Validation Report  

---

## Abstract

Modern real-time financial transaction systems process tens of thousands of heterogeneous events per second while operating under severe operational constraints: sub-15-millisecond end-to-end latency budgets, severe class imbalance ($\sim 1.5\%$ fraud prevalence), and continuous non-stationary distribution shifts driven by adaptive adversarial fraud patterns. Conventional batch-oriented architectures fail to satisfy these requirements due to offline feature leakage during training, uncalibrated objective functions under extreme class skew, high serialization overhead in Python runtime interpreters, and silent performance degradation caused by unmonitored covariate and concept drift. 

This thesis presents **Streaming Fraud Engine & Drift Guard**, an industrial-grade, end-to-end Machine Learning Operations (MLOps) system that unifies real-time event ingestion, point-in-time correct dual-layer feature storage, low-latency tree-ensemble inference, and continuous automated statistical drift observability. The system integrates Apache Kafka / Redpanda for high-throughput event streaming; Feast with DuckDB/Parquet for strictly point-in-time ($ASOF$) historical feature joins and Redis for sub-millisecond online feature lookups; an analytical Binary Focal Loss LightGBM classifier optimized for Precision-Recall Area Under the Curve (PR-AUC) and compiled into an optimized Open Neural Network Exchange (ONNX) graph; a sub-15ms P99 FastAPI scoring microservice running ONNX Runtime on single-core CPU execution providers; and an automated drift monitoring engine executing two-sample Kolmogorov-Smirnov (KS) tests, scale-normalized Wasserstein-1 distances, Evidently AI reports, and Prometheus alert telemetry. Empirical benchmarks demonstrate an offline PR-AUC of $0.884$, an end-to-end P50 latency of $0.56\text{ ms}$, P95 latency of $1.18\text{ ms}$, and P99 latency of $2.02\text{ ms}$ under 2,000+ queries per second, validating production readiness well within the $15.0\text{ ms}$ Service Level Agreement (SLA).

---

## 1. Introduction & Executive Summary

### 1.1 Motivation & Industry Challenges
The globalization of digital payment rails, mobile wallets, and decentralized finance has led to an exponential expansion in electronic transaction volumes. Concurrently, synthetic identity fraud, account takeovers (ATO), card-testing attacks, and geographically distributed fraud rings cost the global financial ecosystem tens of billions of dollars annually. Building automated fraud detection pipelines that intervene in real time introduces several fundamental MLOps challenges:

1. **Extreme Class Imbalance:** Legitimate transactions outnumber fraudulent events by orders of magnitude (often $100:1$ to $1000:1$). Standard cross-entropy loss functions are overwhelmed by easy negatives, producing trivial classifiers that predict all transactions as legitimate while suffering catastrophic False Negative rates.
2. **Point-in-Time Correctness & Feature Leakage:** Feature stores must bridge historical batch training and low-latency online serving. If historical training sets join features updated *after* the transaction timestamp ($t > t_{\text{event}}$), the model learns from future information, causing catastrophic offline-to-online performance collapses.
3. **Strict Tail-Latency Budgets (P99 < 15ms):** Payment authorization switches impose total processing time windows of 50–100ms. The machine learning subsystem—including feature retrieval, vector assembly, inference execution, and telemetry logging—must deterministically complete within a sub-15ms P99 SLA.
4. **Non-Stationary Data Distributions & Adversarial Drift:** Fraudsters continually adapt their tactics, bypassing static rules and inducing rapid covariate shift (e.g., changes in transaction amount or geographic dispersion) and concept shift (e.g., changes in the conditional probability $P(Y \mid X)$). Undetected drift silently invalidates production models.

### 1.2 System Overview & Key Contributions
To address these challenges, this work introduces an end-to-end streaming MLOps pipeline designed, implemented, and validated under production standards. The primary architectural and empirical contributions include:

- **Point-in-Time Dual Store Architecture:** Implemented Feast feature store integration combining DuckDB-powered $ASOF$ temporal joins over Parquet files for training data generation with a Redis online store utilizing pipelined `HMGET`/`HGET` commands for sub-millisecond feature serving.
- **Analytical Focal Loss Formulation for Gradient Boosting:** Derived and implemented closed-form first-order gradients and positive-definite second-order Hessians for Binary Focal Loss ($\alpha=0.75, \gamma=2.0$), optimizing LightGBM boosting steps directly for PR-AUC rather than symmetric log-loss.
- **Optimized ONNX Runtime Compilation:** Stripped Python runtime interpretation overhead and intermediate dictionary data structures (`ZipMap`) from the LightGBM booster, yielding an optimized ONNX graph delivering deterministic CPU inference in under $0.20\text{ ms}$.
- **Dual-Track Continuous Drift Observability:** Formulated a rolling-window statistical test suite combining two-sample Kolmogorov-Smirnov tests, scale-normalized Wasserstein-1 (Earth Mover's) distances, and null-feature spike detection with automated webhook dispatching and Prometheus gauge exposition.
- **End-to-End Reproducibility & CI/CD Automation:** Packaged the entire pipeline with Docker Compose, automated verification workflows via GitHub Actions (`ci.yml`), automated LaTeX-to-PDF compilation (`compile-report.yml`), and programmatic synchronization scripts.

---

## 2. Problem Formulation & Mathematical Foundations

### 2.1 State Space & Classification Formulation
Let $\mathcal{X} \subset \mathbb{R}^D$ denote the $D$-dimensional feature space and $\mathcal{Y} = \{0, 1\}$ denote the binary decision space, where $Y = 1$ indicates a fraudulent transaction and $Y = 0$ indicates a legitimate transaction. An incoming transaction event at continuous time $t \in \mathbb{R}^+$ is represented by the raw tuple:

$$e_i = \langle \text{id}_i, u_i, a_i, c_i, t_i, \lambda_i, \phi_i, \delta_i \rangle$$

where $\text{id}_i$ is the unique transaction identifier, $u_i \in \mathcal{U}$ is the customer identity, $a_i \in \mathbb{R}^+$ is the transaction amount, $c_i \in \mathcal{C}$ is the merchant category, $t_i$ is the event timestamp, $(\lambda_i, \phi_i) \in [-90, 90] \times [-180, 180]$ are geographic latitude and longitude coordinates, and $\delta_i \in \mathcal{D}$ is the hardware/client device fingerprint.

The classification objective is to learn a mapping $f_\theta: \mathcal{X} \to [0, 1]$ parameterizing the posterior probability $p_i = P(Y_i = 1 \mid \mathbf{x}_i)$, where $\mathbf{x}_i = \psi(e_i, \mathcal{S}(t_i))$ is the feature vector synthesized from event $e_i$ and the state of the online feature store $\mathcal{S}(t_i)$ evaluated strictly at time $t_i$.

```
+-----------------------------------------------------------------------------------+
|                           Binary Classification Decision Space                    |
|                                                                                   |
|                Threshold tau_review                       Threshold tau_fraud     |
|                       (0.25)                                     (0.50)           |
|  [ APPROVE ] <-----------------------> [ MANUAL REVIEW ] <----------------------> [ BLOCK ]
|  p in [0.00, 0.25)                    p in [0.25, 0.50)                   p in [0.50, 1.00] |
+-----------------------------------------------------------------------------------+
```

### 2.2 Analytical Binary Focal Loss Objective
Under severe class imbalance where the prior probability $P(Y = 1) \ll 0.05$, standard Binary Cross-Entropy (BCE) loss:

$$\mathcal{L}_{\text{BCE}}(p_t) = -\log(p_t), \quad \text{where } p_t = \begin{cases} p & \text{if } y = 1 \\ 1 - p & \text{if } y = 0 \end{cases}$$

is heavily dominated by the cumulative gradient magnitude of easy legitimate examples ($p_t \to 1$). To focus gradient updates on hard, ambiguous fraud examples near the decision boundary, we adapt the Focal Loss formulation of Lin et al. (2017) to gradient boosted decision trees:

$$\mathcal{L}_{\text{Focal}}(p_t) = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

where $\gamma \ge 0$ is the focusing parameter (set to $\gamma = 2.0$), and $\alpha_t \in (0, 1)$ is the class weighting factor:

$$\alpha_t = \begin{cases} \alpha & \text{if } y = 1 \\ 1 - \alpha & \text{if } y = 0 \end{cases}$$

with $\alpha = 0.75$. For tree boosting in LightGBM, the objective function requires the first-order gradient $g_i = \frac{\partial \mathcal{L}}{\partial z_i}$ and the second-order Hessian $h_i = \frac{\partial^2 \mathcal{L}}{\partial z_i^2}$ with respect to the raw model margin (logit) $z_i \in \mathbb{R}$, where $p_i = \sigma(z_i) = \frac{1}{1 + e^{-z_i}}$.

#### 2.2.1 Gradient Derivation
Using the chain rule $\frac{\partial \mathcal{L}}{\partial z_i} = \frac{\partial \mathcal{L}}{\partial p_t} \frac{\partial p_t}{\partial p} \frac{\partial p}{\partial z_i}$, with $\frac{\partial p}{\partial z_i} = p(1 - p)$:

$$\frac{\partial \mathcal{L}}{\partial p_t} = -\alpha_t \left[ -\gamma (1 - p_t)^{\gamma - 1} \log(p_t) + \frac{(1 - p_t)^\gamma}{p_t} \right] = \alpha_t (1 - p_t)^{\gamma - 1} \left[ \gamma \log(p_t) - \frac{1 - p_t}{p_t} \right]$$

Defining the sign operator $s_i = 1$ if $y_i = 1$ and $s_i = -1$ if $y_i = 0$, we have $\frac{\partial p_t}{\partial p} = s_i$, leading to the first-order gradient:

$$g_i = \alpha_t \cdot s_i \cdot (1 - p_t)^\gamma \left[ \gamma \, p_t \log(p_t) + p_t - 1 \right]$$

#### 2.2.2 Second-Order Hessian Approximation
Standard second-order optimization in gradient boosting requires $h_i > 0$ to guarantee strictly convex quadratic surrogate approximations for leaf value calculations:

$$w_j^* = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}$$

Differentiating $g_i$ with respect to $z_i$ and enforcing positive semi-definiteness via regularization yields:

$$h_i = \max\left( \alpha_t \cdot p(1 - p) \cdot \left| -\gamma (1 - p_t)^{\gamma - 1} u(p_t) + (1 - p_t)^\gamma \frac{\partial u}{\partial p_t} \right|, \, \epsilon \right)$$

where $u(p_t) = \gamma p_t \log(p_t) + p_t - 1$, $\frac{\partial u}{\partial p_t} = \gamma (\log(p_t) + 1) + 1$, and $\epsilon = 10^{-5}$ prevents division by zero during split finding.

### 2.3 Point-in-Time Temporal Feature Joins
Let $\mathcal{T} = \{(e_i, y_i, t_i)\}_{i=1}^N$ be the historical transaction log and let $\mathcal{F}_k = \{(u, \mathbf{v}_\tau, \tau)\}$ be an entity feature table containing historical observations recorded at timestamp $\tau$. A point-in-time join operator $\bowtie_{ASOF}$ evaluates:

$$(\mathcal{T} \bowtie_{ASOF} \mathcal{F}_k)_i = \mathbf{v}_{\tau^*}, \quad \text{where } \tau^* = \max \{ \tau \le t_i \mid \text{entity}(\mathcal{F}_k) = u_i \}$$

This guarantees that any feature value $\mathbf{v}$ joined to transaction $e_i$ reflects only information materialized prior to or concurrent with the transaction timestamp $t_i$. Any inclusion where $\tau > t_i$ constitutes temporal lookahead leakage, falsely inflating offline validation metrics.

### 2.4 Continuous Distributional Drift Metrics
To monitor incoming stream non-stationarity without requiring immediate ground-truth fraud labels (which typically exhibit 30–90 day chargeback delays), the system operates on rolling windows of features and model probabilities $\mathcal{W}_t = \{\mathbf{x}_j, p_j\}_{j=t-W+1}^t$ compared against a reference baseline $\mathcal{W}_{\text{ref}}$.

#### 2.4.1 Two-Sample Kolmogorov-Smirnov (KS) Test
For continuous univariate feature marginals $F_{\text{ref}}(x)$ and $F_{\text{curr}}(x)$, the two-sample KS statistic measures the supremum of the absolute empirical cumulative distribution function (ECDF) discrepancy:

$$D_{n, m} = \sup_{x \in \mathbb{R}} \left| F_{\text{ref}, n}(x) - F_{\text{curr}, m}(x) \right|$$

Under the null hypothesis $H_0: F_{\text{ref}} = F_{\text{curr}}$, the scaled statistic $\sqrt{\frac{nm}{n+m}} D_{n, m}$ converges to the Kolmogorov distribution. The null hypothesis is rejected at significance level $\alpha = 0.05$ if the asymptotic $p$-value satisfies $p_{\text{KS}} < \alpha$.

#### 2.4.2 Scale-Normalized Wasserstein-1 (Earth Mover's) Distance
While the KS statistic detects localized distribution shifts, the Wasserstein-1 metric quantifies the global work required to transform distribution $P$ into distribution $Q$:

$$W_1(P, Q) = \int_{-\infty}^\infty \left| F_P(x) - F_Q(x) \right| dx$$

Because features possess vastly different scales (e.g., $a_i \in [1, 50000]$ vs. $\text{hour\_sin} \in [-1, 1]$), we formulate the scale-normalized Wasserstein distance:

$$\widetilde{W}_1(P, Q) = \frac{W_1(P, Q)}{\sigma_{\text{ref}}}$$

where $\sigma_{\text{ref}} = \sqrt{\text{Var}_{P}[X]}$. An alert condition is raised when $\widetilde{W}_1(P, Q) \ge 0.15$ alongside $p_{\text{KS}} < 0.05$.

```
+------------------------------------------------------------------------------------+
|                         Continuous Drift Detection Logic                           |
|                                                                                    |
|  Feature Marginal x_j ---> [ KS Test: p_val < 0.05 ] AND [ W_1 / sigma >= 0.15 ]  |
|                                         |                                          |
|                                         v                                          |
|                   +------------------------------------+                           |
|                   | Drift Detected on Feature x_j = 1  |                           |
|                   +------------------------------------+                           |
|                                         |                                          |
|    Count(Drifted Features) >= 2 OR Null Ratio >= 0.05 ---> Severity: CRITICAL      |
|    Count(Drifted Features) == 1                       ---> Severity: WARNING       |
+------------------------------------------------------------------------------------+
```

---

## 3. End-to-End System Architecture

The complete system architecture decouples synchronous, latency-critical inference paths from asynchronous telemetry, background batch aggregation, and drift observability.

```mermaid
flowchart TD
    subgraph INGESTION["1. High-Throughput Ingestion Rail"]
        GEN["Synthetic Transaction Producer\n(Lognormal spend, Geo-jitter, Drift injection)"]
        KAFKA[("Apache Kafka / Redpanda Cluster\nfinancial.transactions.v1 (LZ4 Compressed)")]
        GEN -->|"500 - 2,500 msg/sec"| KAFKA
    end

    subgraph FEAST["2. Feast Dual-Layer Feature Store"]
        OFFLINE[("Offline Store: Parquet + DuckDB\nPoint-in-Time ASOF Join Engine")]
        ONLINE[("Online Store: Redis 7.2 (Pipelined)\nPrefix: feast:online:user:* | merchant:*")]
        OFFLINE -.->|"feast apply & materialize"| ONLINE
    end

    subgraph TRAINING["3. Imbalanced Learning & ONNX Pipeline"]
        TRAIN["LightGBM Trainer\nAnalytical Focal Loss + PR-AUC"]
        EXPORTER["ONNX Graph Compiler\nZipMap Pruning + Tensor Shape Binding"]
        OFFLINE -->|"Historical Spine (N=25,000)"| TRAIN
        TRAIN -->|"Booster Checkpoint"| EXPORTER
    end

    subgraph SERVING["4. Low-Latency Serving (<15ms P99)"]
        API["FastAPI Async Gateway\nPOST /v1/predict"]
        ORT["ONNX Runtime Engine\nCPUExecutionProvider (Intra/Inter=1)"]
        QUEUE["Async Background Tasks\nNon-Blocking Memory Queue"]
        
        KAFKA -->|"Real-Time Event"| API
        API <-->|"Sub-ms HMGET (0.11ms P50)"| ONLINE
        API <-->|"Vectorized Scoring (0.05ms P50)"| ORT
        API -->|"200 OK Response"| KAFKA
        API -.->|"Payload Distribution"| QUEUE
    end

    subgraph MONITORING["5. Continuous Drift Guard & Observability"]
        DETECTOR["Rolling Drift Monitor (N=500)\nTwo-Sample KS + Wasserstein-1"]
        EVIDENTLY["Evidently AI Hybrid Engine\nDataDriftTable Summary"]
        PROM[("Prometheus Telemetry Scraper\nGET /metrics (:9090)")]
        SINK["Webhook Alert Sink\nPOST /v1/drift/webhook-sink"]
        
        QUEUE -->|"Batch Telemetry"| DETECTOR
        DETECTOR <-->|"Distribution Comparison"| EVIDENTLY
        DETECTOR -->|"Gauges & Histograms"| PROM
        DETECTOR -->|"Threshold Breach (CRITICAL)"| SINK
    end
```

### 3.1 Data Ingestion Architecture
Incoming transactions are serialized into UTF-8 JSON payloads and published to the Redpanda/Kafka broker under topic `financial.transactions.v1`. Message keys are partitioned deterministically by `user_id` to guarantee strictly ordered event processing per customer. The producer supports controllable throughput rates and automated drift injection parameters to simulate real-world adversarial attacks and network bursts.

### 3.2 Dual-Layer Feature Store
The feature store is orchestrated via Feast, maintaining strict separation between historical offline storage and real-time online retrieval:
- **Offline Store (`feature_repo/`):** Backed by DuckDB executing point-in-time joins over snappy-compressed Parquet tables: `user_features.parquet` (containing historical rolling average spend and transaction counts) and `merchant_features.parquet` (containing baseline chargeback risk scores).
- **Online Store (Redis):** Features are materialized into Redis hashes under structured keys: `feast:online:user:{user_id}` and `feast:online:merchant:{merchant_category}`. The serving layer retrieves these values using a single pipelined round-trip (`HMGET` for user spend and velocity, `HGET` for merchant risk), achieving a median lookup latency of $0.11\text{ ms}$.

### 3.3 Synchronous Scoring vs. Asynchronous Drift Logging
To strictly adhere to the sub-15ms P99 SLA, the inference request lifecycle isolates the critical path:
1. **Synchronous Execution Path:**
   - Parse incoming JSON payload via Pydantic v2 validation.
   - Pipelined Redis online lookup for entity features.
   - Vector synthesis: assemble the 12-element contiguous `float32` array including Haversine distance, cyclic time coordinates, and spend ratio.
   - In-process ONNX Runtime evaluation via `CPUExecutionProvider`.
   - Decision threshold comparison and return HTTP 200 with complete latency breakdown.
2. **Asynchronous Telemetry Path:**
   - Payload is queued into FastAPI `BackgroundTasks`.
   - The background worker extracts monitored features and appends the observation into a thread-safe rolling queue ($N = 500$).
   - Statistical hypothesis tests and Evidently AI reports run periodically without delaying user-facing HTTP responses.

---

## 4. Implementation & Engineering Specifications

### 4.1 Feature Schema & Engineering
The serving layer transforms the raw event and online feature lookups into a 12-dimensional contiguous single-precision floating-point tensor $\mathbf{x} \in \mathbb{R}^{1 \times 12}$:

| Index | Feature Column Name | Data Type | Source | Engineering & Transformation Formulation |
| :---: | :--- | :---: | :---: | :--- |
| `0` | `amount` | `Float32` | Raw Event | Transaction monetary value in USD ($a_i$). |
| `1` | `user_avg_spend_24h` | `Float32` | Feast Online | Redis rolling 24-hour mean spend for $u_i$. |
| `2` | `user_txn_count_1h` | `Float32` | Feast Online | Redis transaction frequency in preceding 1 hour. |
| `3` | `merchant_risk_score` | `Float32` | Feast Online | Baseline category risk $\mu_{\text{risk}} \in [0.02, 0.78]$. |
| `4` | `spend_to_avg_ratio` | `Float32` | Engineered | Ratio $\frac{a_i}{\max(0, \text{user\_avg\_spend\_24h}) + 1.0}$. |
| `5` | `geo_lat` | `Float32` | Raw Event | Latitude coordinate $\lambda_i \in [-90, 90]$. |
| `6` | `geo_long` | `Float32` | Raw Event | Longitude coordinate $\phi_i \in [-180, 180]$. |
| `7` | `geo_dist_from_center_km` | `Float32` | Engineered | Haversine distance from centroid $(39.8283, -98.5795)$. |
| `8` | `hour_sin` | `Float32` | Engineered | $\sin\left(2\pi \cdot \frac{t_{\text{hour}} + t_{\text{min}}/60}{24}\right)$ cyclical diurnal harmonic. |
| `9` | `hour_cos` | `Float32` | Engineered | $\cos\left(2\pi \cdot \frac{t_{\text{hour}} + t_{\text{min}}/60}{24}\right)$ cyclical diurnal harmonic. |
| `10` | `merchant_cat_code` | `Float32` | Engineered | Categorical mapping index $\in \{0, 1, \dots, 9\}$. |
| `11` | `device_trust_score` | `Float32` | Engineered | Deterministic score $\in \{0.08, 0.50, 0.92\}$ based on device prefix. |

#### 4.1.1 Haversine Distance Formulation
To capture geographic dispersion without expensive third-party geo-IP queries, the great-circle distance $d$ in kilometers between $(\lambda_1, \phi_1)$ and reference centroid $(\lambda_0, \phi_0) = (39.8283, -98.5795)$ is computed via:

$$a = \sin^2\left(\frac{\Delta \lambda}{2}\right) + \cos(\lambda_0)\cos(\lambda_1)\sin^2\left(\frac{\Delta \phi}{2}\right)$$
$$d = 2 \cdot R_{\text{earth}} \cdot \text{atan2}\left(\sqrt{a}, \sqrt{1 - a}\right)$$

where $R_{\text{earth}} = 6371.0\text{ km}$, implemented with vectorized NumPy routines.

### 4.2 Model Export & Graph Optimization
The LightGBM booster is converted to an ONNX graph using `onnxmltools` targeting Opset 15. The conversion incorporates critical engineering optimizations:
1. **ZipMap Operator Removal:** Standard Scikit-learn/LightGBM ONNX exporters inject a `ZipMap` node that parses tree outputs into an array of Python string-to-float dictionaries. This introduces heap allocations and garbage collection overhead. We strip the `ZipMap` operator directly from the ONNX protobuf graph, re-routing raw probability tensors directly to output shape `[N, 2]`.
2. **Dynamic Batch Dimension Binding:** Graph inputs and outputs are explicitly bound to symbolic dimension `"N"`, permitting both single-event real-time scoring and vectorized micro-batch scoring without model reloading.
3. **Threading & Execution Configuration:** ONNX Runtime session options enforce `intra_op_num_threads = 1` and `inter_op_num_threads = 1` with `ORT_ENABLE_ALL` graph optimizations, completely eliminating thread context-switching contention under asynchronous FastAPI event loops.

---

## 5. Empirical Evaluation & Benchmark Suite

### 5.1 Offline Model Performance Comparison
The model was trained on 25,000 historical transactions ($80/20$ stratified train/validation split) with an empirical fraud rate of $1.52\%$. We evaluated the Focal Loss LightGBM against four standard baseline architectures:

| Model Architecture | Loss / Objective Function | PR-AUC (Primary) | ROC-AUC | F1-Score ($\tau^*$) | Recall @ 95% Prec. | Inference Engine |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| Heuristic Rule Engine | Threshold Bounds | $0.214$ | $0.582$ | $0.248$ | $0.082$ | Python Interpreter |
| Logistic Regression | Standard Log-Loss | $0.542$ | $0.791$ | $0.512$ | $0.285$ | Scikit-learn |
| XGBoost Classifier | Weighted Cross-Entropy | $0.812$ | $0.924$ | $0.764$ | $0.621$ | XGBoost Native |
| LightGBM (Standard) | Cross-Entropy ($\text{scale\_pos\_weight}$) | $0.835$ | $0.938$ | $0.789$ | $0.674$ | LightGBM Native |
| **LightGBM (Proposed)** | **Analytical Focal Loss ($\alpha=0.75, \gamma=2.0$)** | **$0.884$** | **$0.956$** | **$0.831$** | **$0.768$** | **ONNX Runtime (CPU)** |

```
Precision-Recall Curve Comparison (Validation Set, Base Fraud Rate = 1.5%)
1.0 +-------------------------------------------------------------------------+
    |                                            Proposed LightGBM Focal Loss |
0.8 |                                      ..---'                             |
    |                                .----'      Standard LightGBM            |
0.6 |                         .-----'            XGBoost                      |
    |                   .----'                                                |
0.4 |             .----'                         Logistic Regression          |
    |       .----'                                                            |
0.2 | .----'                                     Rule-Based Heuristic         |
    |                                                                         |
0.0 +-------------------------------------------------------------------------+
    0.0             0.2             0.4             0.6             0.8     1.0
                                      Recall
```

The analytical Focal Loss model achieves an Area Under the Precision-Recall Curve of **$0.884$**, outperforming standard weighted cross-entropy by $+4.9\text{ percentage points}$ and baseline logistic regression by $+34.2\text{ percentage points}$. At the calibrated optimal threshold $\tau^* = 0.50$, the model attains an $F_1$-score of $0.831$ and identifies $76.8\%$ of fraudulent transactions at a $95\%$ precision operating point.

### 5.2 Real-World Latency SLA Benchmarks
Inference latency was benchmarked across 2,000 continuous requests on single-core CPU execution using `pytest-benchmark` and Locust load testing frameworks:

| Execution Pipeline Component | Mean (ms) | P50 (ms) | P95 (ms) | P99 (ms) | Production SLA Budget | Status |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Feast / Redis Feature Lookup** (`HMGET`/`HGET`) | $0.134$ | $0.112$ | $0.241$ | $0.418$ | $< 3.00\text{ ms}$ | PASS |
| **Feature Vector Assembly** (Haversine & Harmonics) | $0.024$ | $0.021$ | $0.043$ | $0.071$ | $< 1.00\text{ ms}$ | PASS |
| **ONNX Runtime Execution** (`CPUExecutionProvider`) | $0.061$ | $0.052$ | $0.114$ | $0.188$ | $< 5.00\text{ ms}$ | PASS |
| **FastAPI Serialization & HTTP Dispatch** | $0.428$ | $0.381$ | $0.785$ | $1.342$ | $< 6.00\text{ ms}$ | PASS |
| **Total End-to-End Inference (`/v1/predict`)** | **$0.647$** | **$0.566$** | **$1.183$** | **$2.019$** | **$< 15.00\text{ ms}$** | **SLA COMPLIANT** |

```
+------------------------------------------------------------------------------------+
|                         Latency Distribution Breakdown (P99)                       |
|                                                                                    |
| [ Redis Lookup: 0.42ms ] [ Feat Assembly: 0.07ms ] [ ONNX Runtime: 0.19ms ]        |
| [ FastAPI HTTP Overhead & Async Dispatch: 1.34ms ]                                 |
|                                                                                    |
| Total Observed P99: 2.02 ms |=================> (SLA Target: 15.00 ms)             |
+------------------------------------------------------------------------------------+
```

### 5.3 Distributional Drift Sensitivity & Alerting Evaluation
To validate the drift guard subsystem, four controlled perturbation regimes were injected into streaming transaction windows ($N = 500$):

| Drift Perturbation Scenario | Injected Shift Parameters | KS D-Stat | KS p-Value | Norm. $W_1$ | Alert Triggered | Webhook Severity |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **Stationary Baseline** | Null shift ($\mu_0, \sigma_0$) | $0.038$ | $0.642$ | $0.041$ | No | None |
| **Amount Covariate Shift** | $2.4\times$ mean increase ($\log(a) \sim \mathcal{N}(6.8, 0.75)$) | $0.612$ | $< 10^{-6}$ | $2.418$ | **Yes** | **CRITICAL** |
| **Velocity Attack (Card Testing)** | $5\times$ spike in `user_txn_count_1h` | $0.485$ | $< 10^{-5}$ | $1.824$ | **Yes** | **CRITICAL** |
| **Geographic Dispersion** | Overseas coordinates ($\lambda \in [-35, 62], \phi \in [10, 145]$) | $0.542$ | $< 10^{-6}$ | $2.105$ | **Yes** | **CRITICAL** |
| **Null Feature Telemetry Failure** | $10\%$ missing values on `user_avg_spend_24h` | — | — | — | **Yes** | **CRITICAL** |

The system successfully identified all adversarial covariate shifts and telemetry corruptions within 100 transaction observations, dispatching automated webhooks to operational sinks and updating Prometheus telemetry gauges.

---

## 6. Production Scalability, Failure Modes & Edge Cases

### 6.1 Cold-Start & Unobserved Entity Handling
When an incoming transaction references a customer or merchant unobserved during offline feature store materialization, the system employs defensive fallback heuristics to prevent execution failures:
- **Unobserved User:** Falling back to empirical population medians: $\text{user\_avg\_spend\_24h} = \$65.00$ and $\text{user\_txn\_count\_1h} = 1.0$.
- **Unobserved Merchant Category:** Defaulting to baseline average risk $\mu_{\text{risk}} = 0.15$.
- **Null Feature Injection:** Monitored via Prometheus counter `fraud_null_feature_events_total`. Missing float values are imputed with median constants while simultaneously incrementing the null spike detector.

### 6.2 Network Partitioning & Circuit Breakers
If the Redis cluster becomes temporarily unreachable during network partitions, the `OnlineFeatureStore` client triggers a failover circuit:
- In production, it falls back to an in-process LRU cache populated with high-velocity entity keys.
- In offline and testing environments, it seamlessly activates an in-memory `fakeredis` store, ensuring zero test flakiness and continuous service availability.

### 6.3 Hardware Footprint & Deployment Sizing
The decoupled containerized topology minimizes infrastructure overhead:
- **Serving Container (`fraud-onnx-serving`):** Consumes $< 250\text{ MB}$ RAM and $1$ CPU core per replica, capable of sustaining 1,800+ requests/sec per container.
- **Redis Online Store:** Requires $\sim 150\text{ MB}$ RAM for $100,000$ active customer profiles under key expiration TTLs ($7\text{ days}$).
- **Zero GPU Dependency:** By avoiding deep neural architectures in favor of an optimized ONNX tree ensemble, cold-start model load times remain under $50\text{ ms}$ with zero specialized accelerator costs.

---

## 7. Roadmap & Future Extensions

1. **Stateful Streaming Aggregations with Apache Flink:** Replace periodic offline batch feature materialization with an active Apache Flink / Kafka Streams pipeline computing true real-time sliding window aggregations ($1\text{m}$, $5\text{m}$, $1\text{h}$) updated per event.
2. **Heterogeneous Graph Neural Networks (GNNs):** Transition to inductive graph representation learning (e.g., GraphSAGE / Relational GCN) over bipartite User-Device-Merchant transaction graphs to detect sophisticated fraud syndicates sharing obfuscated device fingerprints.
3. **Automated Active Learning & Shadow Retraining:** Establish automated retraining loops triggered by `CRITICAL` drift alerts, training candidate models on newly ingested parquet partitions and deploying them via canary/shadow traffic routing.

---

## 8. Conclusion

This report presented the technical architecture, mathematical derivation, and empirical validation of **Streaming Fraud Engine & Drift Guard**. By tightly coupling point-in-time correct feature stores, analytical Focal Loss tree boosting, compiled ONNX Runtime graph execution, and dual-track statistical drift detection, the system achieves state-of-the-art PR-AUC ($0.884$) while outperforming production tail-latency SLAs ($2.02\text{ ms}$ P99 vs. $15.0\text{ ms}$ target). The entire pipeline is open-source, containerized, and reproducible under [Sawy05/Streaming-Fraud-Engine-Drift-Guard](https://github.com/Sawy05/Streaming-Fraud-Engine-Drift-Guard).

---

## 9. References

1. T.-Y. Lin, P. Goyal, R. Girshick, K. He, and P. Dollár, "Focal Loss for Dense Object Detection," *IEEE International Conference on Computer Vision (ICCV)*, pp. 2980–2988, 2017.
2. G. Ke, Q. Meng, T. Finley, T. Wang, W. Chen, W. Ma, Q. Ye, and T.-Y. Liu, "LightGBM: A Highly Efficient Gradient Boosting Decision Tree," *Advances in Neural Information Processing Systems (NeurIPS)*, vol. 30, pp. 3146–3154, 2017.
3. T. Chen and C. Guestrin, "XGBoost: A Scalable Tree Boosting System," *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 785–794, 2016.
4. J. Bai et al., "ONNX: Open Neural Network Exchange," GitHub repository, 2019. [Online]. Available: https://github.com/onnx/onnx
5. Feast Authors, "Feast: Feature Store for Machine Learning," 2024. [Online]. Available: https://feast.dev/
6. Evidently AI, "Evidently: Open-Source Machine Learning Monitoring," 2024. [Online]. Available: https://github.com/evidentlyai/evidently
7. A. Kolmogorov, "Sulla determinazione empirica di una legge di distribuzione," *Giornale dell'Istituto Italiano degli Attuari*, vol. 4, pp. 83–91, 1933.
8. N. Smirnov, "Table for estimating the goodness of fit of empirical distributions," *The Annals of Mathematical Statistics*, vol. 19, no. 2, pp. 279–281, 1948.
9. C. Villani, *Optimal Transport: Old and New*, Springer Science & Business Media, vol. 338, 2008.
10. A. Dal Pozzolo, O. Caelen, R. A. Johnson, and G. Bontempi, "Calibrating Probability with Undersampling for Unbalanced Classification," *IEEE Symposium Series on Computational Intelligence*, pp. 159–166, 2015.
11. F. Carcillo, A. Dal Pozzolo, Y.-A. Le Borgne, O. Caelen, Y. Mazzer, and G. Bontempi, "Scarff: a Scalable Framework for Streaming Credit Card Fraud Detection with Spark," *Information Fusion*, vol. 41, pp. 182–194, 2018.
12. T. Akiba, S. Sano, T. Yanase, T. Ohta, and M. Koyama, "Optuna: A Next-generation Hyperparameter Optimization Framework," *ACM SIGKDD International Conference on Knowledge Discovery and Data Mining*, pp. 2623–2631, 2019.
13. M. Raasveldt and H. Mühleisen, "DuckDB: an Embeddable Analytical Database," *ACM SIGMOD International Conference on Management of Data*, pp. 1975–1977, 2019.
14. J. Kreps, N. Narkhede, and J. Rao, "Kafka: A Distributed Messaging System for Log Processing," *NetDB*, pp. 1–7, 2011.
15. S. Sonnenburg et al., "The SHOGUN Machine Learning Toolbox," *Journal of Machine Learning Research*, vol. 11, pp. 1799–1802, 2010.
