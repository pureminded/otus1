# Архитектура и декомпозиция антифрод-системы

## 1. Общая архитектура системы

### 1.1 High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        КЛИЕНТЫ БАНКА                             │
│                    (Онлайн транзакции)                           │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│              СУЩЕСТВУЮЩАЯ ПЛАТЕЖНАЯ СИСТЕМА БАНКА                │
│                    (Transaction Gateway)                         │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ CSV Batch / Real-time Stream
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                   АНТИФРОД ML-СИСТЕМА                            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  1. DATA INGESTION LAYER                                │   │
│  │     - CSV Parser                                        │   │
│  │     - Data Validation                                   │   │
│  │     - Feature Store                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  2. INFERENCE LAYER                                     │   │
│  │     - Feature Engineering Pipeline                      │   │
│  │     - Model Serving (REST API)                          │   │
│  │     - Decision Engine                                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  3. TRAINING LAYER (offline)                            │   │
│  │     - Data Preparation                                  │   │
│  │     - Model Training Pipeline                           │   │
│  │     - Hyperparameter Tuning                             │   │
│  │     - Model Validation                                  │   │
│  └─────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  4. MONITORING & FEEDBACK LAYER                         │   │
│  │     - Metrics Collection                                │   │
│  │     - Drift Detection                                   │   │
│  │     - Alerting System                                   │   │
│  │     - Feedback Loop                                     │   │
│  └─────────────────────────────────────────────────────────┘   │
└───────────────────────────┬─────────────────────────────────────┘
                            │
                            │ Predictions & Metrics
                            ▼
┌─────────────────────────────────────────────────────────────────┐
│                 ВНУТРЕННИЕ СИСТЕМЫ БАНКА                         │
│  - Fraud Analysts Dashboard                                      │
│  - Customer Support System                                       │
│  - Reporting & Analytics                                         │
└─────────────────────────────────────────────────────────────────┘
```

### 1.2 Архитектурные принципы

1. **Модульность**: каждый компонент - независимый модуль с четким API
2. **Масштабируемость**: горизонтальное масштабирование критичных компонентов
3. **Отказоустойчивость**: отсутствие single point of failure
4. **Observability**: полное логирование и мониторинг всех компонентов
5. **Security by design**: безопасность на каждом уровне архитектуры

---

## 2. Декомпозиция системы на функциональные модули

### 2.1 Data Ingestion Layer (Слой приема данных)

#### Назначение
Прием, валидация и предварительная обработка входящих данных о транзакциях.

#### Компоненты

**2.1.1 CSV Parser Service**
- **Функции**:
  - Чтение CSV файлов из хранилища
  - Парсинг строк в структурированный формат
  - Обработка различных форматов дат, кодировок
- **Технологии**: Python, pandas, dask (для больших файлов)
- **Input**: CSV files from S3/Cloud Storage
- **Output**: Structured DataFrames

**2.1.2 Data Validation Service**
- **Функции**:
  - Проверка схемы данных (типы, обязательные поля)
  - Валидация бизнес-правил (сумма > 0, валидные даты)
  - Детектирование аномалий в данных
  - Генерация отчетов о качестве данных
- **Технологии**: Great Expectations, Pandera
- **Input**: Structured DataFrames
- **Output**: Validated data + quality reports

**2.1.3 Feature Store**
- **Функции**:
  - Хранение предвычисленных признаков
  - Обслуживание признаков для training и inference
  - Версионирование feature schemas
  - Управление Feature lineage
- **Технологии**: Feast, Hopsworks (опционально)
- **Storage**: PostgreSQL / Redis для online features

**2.1.4 Data Enrichment Service**
- **Функции**:
  - Добавление внешних признаков (geo-данные, device fingerprint)
  - Агрегация исторических данных клиента
  - Расчет derived features
- **Input**: Validated transactions
- **Output**: Enriched feature vectors

---

### 2.2 Inference Layer (Слой инференса)

#### Назначение
Предсказание мошенничества для новых транзакций в режиме реального времени.

#### Компоненты

**2.2.1 Feature Engineering Pipeline**
- **Функции**:
  - Преобразование raw данных в признаки модели
  - Применение feature transformations (scaling, encoding)
  - Расчет real-time агрегаций (если требуется)
- **Технологии**: scikit-learn Pipeline, feature-engine
- **Latency requirement**: < 20 мс

**2.2.2 Model Serving API**
- **Функции**:
  - REST API для получения предсказаний
  - Batch prediction endpoint
  - Управление версиями моделей
  - Load balancing между репликами
- **Технологии**: FastAPI, uvicorn
- **Endpoints**:
  - `POST /predict` - single prediction
  - `POST /batch-predict` - batch processing
  - `GET /health` - health check
  - `GET /model-info` - model metadata
- **Latency requirement**: < 100 мс end-to-end

**2.2.3 Model Container**
- **Функции**:
  - Загрузка сериализованной модели
  - Inference на основе feature vector
  - Кэширование модели в памяти
- **Технологии**: pickle, joblib, ONNX (для оптимизации)
- **Models**: XGBoost, LightGBM, CatBoost, или ансамбль

**2.2.4 Decision Engine**
- **Функции**:
  - Преобразование вероятности в решение (ALLOW/BLOCK/REVIEW)
  - Применение бизнес-правил и порогов
  - Risk scoring и categorization
  - Генерация объяснений (SHAP values)
- **Input**: fraud_probability, transaction_features
- **Output**: decision, risk_level, explanation

**2.2.5 Cache Layer**
- **Функции**:
  - Кэширование предсказаний для повторяющихся транзакций
  - Кэширование признаков клиентов
- **Технологии**: Redis
- **TTL**: configurable (например, 5 минут для predictions)

---

### 2.3 Training Layer (Слой обучения)

#### Назначение
Offline обучение, валидация и регистрация моделей машинного обучения.

#### Компоненты

**2.3.1 Data Preparation Pipeline**
- **Функции**:
  - Загрузка исторических данных для обучения
  - Создание train/validation/test splits
  - Feature engineering (аналогично inference pipeline)
  - Обработка дисбаланса классов (SMOTE, undersampling)
- **Технологии**: pandas, scikit-learn, imbalanced-learn
- **Orchestration**: Apache Airflow / Prefect

**2.3.2 Training Pipeline**
- **Функции**:
  - Обучение ML моделей
  - Hyperparameter tuning (Optuna, GridSearch)
  - Cross-validation
  - Ensemble methods
- **Технологии**: scikit-learn, XGBoost, LightGBM
- **Compute**: Cloud VMs with GPU (опционально для NN)

**2.3.3 Model Evaluation Service**
- **Функции**:
  - Расчет метрик на test set (F2-Score, Recall, FPR)
  - Сравнение с baseline и текущей production моделью
  - Генерация evaluation reports
  - Анализ feature importance
- **Output**: metrics, confusion matrix, ROC/PR curves

**2.3.4 Model Registry**
- **Функции**:
  - Версионирование моделей
  - Хранение артефактов моделей (weights, configs, metadata)
  - Статусы моделей (staging, production, archived)
  - Линкование с экспериментами и datasets
- **Технологии**: MLflow, Weights & Biases
- **Storage**: S3-compatible storage

**2.3.5 Automated Retraining Service**
- **Функции**:
  - Scheduled retraining (cron-like)
  - Triggered retraining (при drift)
  - Управление lifecycle моделей
  - Rollback к предыдущим версиям
- **Orchestration**: Airflow DAGs, GitHub Actions

---

### 2.4 Monitoring & Feedback Layer (Слой мониторинга)

#### Назначение
Непрерывный мониторинг качества системы, обнаружение проблем и сбор feedback.

#### Компоненты

**2.4.1 Metrics Collection Service**
- **Функции**:
  - Сбор метрик inference (latency, throughput, error rate)
  - Сбор метрик модели (predictions distribution, confidence scores)
  - Сбор бизнес-метрик (blocked transactions, detected fraud)
- **Технологии**: Prometheus client libraries
- **Storage**: Prometheus TSDB

**2.4.2 Drift Detection Service**
- **Функции**:
  - Детектирование Data Drift (изменения в распределении признаков)
  - Детектирование Concept Drift (изменения в паттернах fraud)
  - Детектирование Model Drift (деградация метрик)
  - Статистические тесты (KS-test, Chi-square)
- **Технологии**: Evidently AI, Alibi Detect
- **Frequency**: hourly checks
- **Output**: drift reports, alerts

**2.4.3 Alerting System**
- **Функции**:
  - Генерация алертов при критических событиях
  - Routing алертов (email, Slack, PagerDuty)
  - Управление правилами алертинга
- **Технологии**: Prometheus Alertmanager
- **Alert conditions**:
  - Recall < 0.95 (warning), < 0.90 (critical)
  - FPR > 0.06 (warning), > 0.08 (critical)
  - Latency > 150 мс (warning), > 200 мс (critical)
  - Drift detected (warning)

**2.4.4 Dashboards & Visualization**
- **Функции**:
  - Real-time dashboards для мониторинга
  - Визуализация метрик ML (Recall, Precision, F2)
  - Визуализация бизнес-метрик (fraud rate, blocked amount)
  - Анализ ошибок (False Positives, False Negatives)
- **Технологии**: Grafana
- **Пользователи**: Data Scientists, MLOps, Fraud Analysts

**2.4.5 Feedback Loop Service**
- **Функции**:
  - Сбор ground truth labels (actual fraud outcomes)
  - Анализ ошибок модели (post-mortem для FP/FN)
  - Обогащение training dataset новыми примерами
  - Приоритизация проблемных случаев для review
- **Input**: predictions + actual outcomes
- **Output**: labeled data for retraining

**2.4.6 Logging & Audit Service**
- **Функции**:
  - Централизованное логирование всех событий
  - Audit trail для всех предсказаний (для compliance)
  - Log analysis для debugging
  - PII masking в логах
- **Технологии**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Retention**: logs 30 дней, audit trail 2 года

---

## 3. Инфраструктурные компоненты

### 3.1 Cloud Infrastructure

**Облачный провайдер**: Yandex Cloud (или AWS/GCP)

**Compute Resources:**
- **Inference API**: 
  - Kubernetes cluster (автоскейлинг 2-10 pods)
  - Instance type: 4 CPU, 8 GB RAM
- **Training Jobs**: 
  - Preemptible VMs (cost optimization)
  - Instance type: 8 CPU, 32 GB RAM (+ GPU опционально)
- **Monitoring**: 
  - Dedicated VM: 2 CPU, 4 GB RAM

**Storage:**
- **Object Storage** (S3-compatible): для CSV files, model artifacts
- **Database** (PostgreSQL): для feature store, audit logs
- **Cache** (Redis): для online feature serving, predictions cache
- **Time-Series DB** (Prometheus): для метрик

### 3.2 CI/CD Pipeline

**Source Control**: GitHub

**CI/CD Tool**: GitHub Actions

**Pipeline stages:**
1. **Code quality checks**: 
   - Linting (pylint, black, mypy)
   - Unit tests (pytest)
   - Code coverage (>80%)
2. **Build**:
   - Docker image build
   - Push to container registry
3. **Deploy to Staging**:
   - Deploy to staging environment
   - Integration tests
   - Smoke tests
4. **Deploy to Production**:
   - Canary deployment (10% traffic)
   - Monitoring for 1 hour
   - Full rollout or rollback

### 3.3 Containerization & Orchestration

**Containerization**: Docker
- Inference API container
- Training job container
- Monitoring container

**Orchestration**: Kubernetes
- Deployment configs
- Services, Ingress
- ConfigMaps, Secrets
- Horizontal Pod Autoscaler (HPA)

### 3.4 Security Components

**Authentication & Authorization:**
- API key authentication для inference API
- RBAC (Role-Based Access Control) в Kubernetes
- Service accounts для межсервисного взаимодействия

**Secrets Management:**
- HashiCorp Vault или Cloud Secrets Manager
- Rotation policy для credentials

**Network Security:**
- VPC с изолированными subnets
- Security groups / Firewalls
- TLS/SSL для всех соединений

**Data Security:**
- Encryption at rest (для storage)
- Encryption in transit (TLS 1.3)
- Data masking для non-production окружений

---

## 4. Data Flow диаграммы

### 4.1 Training Flow (Offline)

```
┌──────────────┐
│ Historical   │
│ Transactions │
│  (CSV files) │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Data Ingestion│ ── Validation ──> Quality Report
│   Pipeline    │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│  Feature     │
│ Engineering  │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Train/Val/   │
│ Test Split   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Model      │ ── Hyperparameter ──> Experiment Tracking
│  Training    │       Tuning            (MLflow)
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Model      │ ── Metrics: F2, ──> Evaluation Report
│ Evaluation   │    Recall, FPR
└──────┬───────┘
       │
       ▼ (if metrics OK)
┌──────────────┐
│   Model      │
│  Registry    │
│ (Staging)    │
└──────┬───────┘
       │
       ▼ (manual approval)
┌──────────────┐
│   Deploy     │
│ to Production│
└──────────────┘
```

### 4.2 Inference Flow (Online)

```
┌──────────────┐
│   Client     │
│ Transaction  │
└──────┬───────┘
       │ POST /predict
       ▼
┌──────────────┐
│  API Gateway │
│  (FastAPI)   │
└──────┬───────┘
       │
       ▼
┌──────────────┐     Cache Hit?
│    Redis     │───────────────────┐
│    Cache     │                   │
└──────┬───────┘                   │
       │ Cache Miss                │
       ▼                           │
┌──────────────┐                   │
│   Feature    │                   │
│ Engineering  │                   │
└──────┬───────┘                   │
       │                           │
       ▼                           │
┌──────────────┐                   │
│    Model     │                   │
│   Inference  │                   │
└──────┬───────┘                   │
       │                           │
       ▼                           │
┌──────────────┐                   │
│   Decision   │                   │
│    Engine    │                   │
└──────┬───────┘                   │
       │                           │
       ├───────────────────────────┘
       │
       ▼
┌──────────────┐
│   Response   │ ─── Log ───> Logging Service
│  (fraud      │
│  prediction) │ ─── Metrics ─> Monitoring
└──────┬───────┘
       │
       ▼
┌──────────────┐
│   Client     │
└──────────────┘
```

### 4.3 Monitoring & Retraining Flow

```
┌──────────────┐
│  Production  │
│ Predictions  │
└──────┬───────┘
       │
       ▼
┌──────────────────────────────────┐
│     Monitoring Service            │
│ ┌─────────┐  ┌────────────────┐ │
│ │ Metrics │  │ Drift Detection│ │
│ └─────────┘  └────────────────┘ │
└───────┬─────────────┬────────────┘
        │             │
        ▼             ▼ (if drift/degradation)
   ┌─────────┐   ┌──────────────┐
   │Dashboard│   │   Alerting   │
   └─────────┘   └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   Trigger    │
                 │  Retraining  │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │   Training   │
                 │   Pipeline   │
                 └──────┬───────┘
                        │
                        ▼
                 ┌──────────────┐
                 │  New Model   │
                 │  (Staging)   │
                 └──────┬───────┘
                        │
                        ▼ (A/B test)
                 ┌──────────────┐
                 │ Gradual      │
                 │ Rollout      │
                 └──────────────┘
```

---

## 5. Technology Stack Summary

### Programming & ML
- **Language**: Python 3.9+
- **ML Libraries**: scikit-learn, XGBoost, LightGBM, CatBoost
- **Feature Engineering**: pandas, numpy, feature-engine
- **Imbalanced Learning**: imbalanced-learn

### Infrastructure
- **Cloud**: Yandex Cloud / AWS / GCP
- **Containers**: Docker
- **Orchestration**: Kubernetes
- **API**: FastAPI + uvicorn
- **Cache**: Redis

### Data Storage
- **Object Storage**: S3-compatible (Cloud Storage)
- **Database**: PostgreSQL
- **Feature Store**: Feast (опционально)
- **Time-Series**: Prometheus

### MLOps
- **Experiment Tracking**: MLflow / Weights & Biases
- **Version Control**: Git (GitHub)
- **Data Versioning**: DVC
- **CI/CD**: GitHub Actions
- **Orchestration**: Apache Airflow / Prefect

### Monitoring & Logging
- **Metrics**: Prometheus
- **Dashboards**: Grafana
- **Logging**: ELK Stack (Elasticsearch, Logstash, Kibana)
- **Drift Detection**: Evidently AI

### Security
- **Secrets**: HashiCorp Vault
- **Encryption**: TLS 1.3
- **Auth**: API Keys, RBAC

---

## 6. Масштабируемость и производительность

### 6.1 Горизонтальное масштабирование

**Inference API:**
- Stateless design → легко масштабируется
- Kubernetes HPA (Horizontal Pod Autoscaler)
- Auto-scaling на основе CPU/memory или custom metrics (RPS)

**Training:**
- Параллельное обучение нескольких моделей
- Distributed training (если модель большая)
- Batch processing для feature engineering

### 6.2 Оптимизация производительности

**Latency optimization:**
- Model compression (pruning, quantization)
- ONNX runtime для ускорения inference
- Caching предсказаний и признаков
- Асинхронная обработка (async/await в FastAPI)

**Throughput optimization:**
- Batch prediction для offline обработки
- Load balancing между репликами API
- Connection pooling для database

### 6.3 Capacity Planning

**Baseline (средняя нагрузка):**
- 50 транзакций/сек
- 2-3 inference pods (по 25 RPS каждый)
- CPU usage: ~40%

**Peak (праздники):**
- 400 транзакций/сек
- Auto-scale до 10-15 pods
- CPU usage: ~70%

**Резерв:** +20% capacity для непредвиденных пиков

---

## 7. Disaster Recovery & High Availability

### 7.1 Availability Strategy

**SLA Target**: 99.9% (43 минуты downtime в месяц)

**Обеспечение:**
- Multi-zone deployment (Kubernetes across AZs)
- Redundancy: минимум 2 replicas для каждого сервиса
- Health checks и automatic restart
- Load balancer с failover

### 7.2 Backup & Recovery

**Model artifacts:**
- Версионирование в Model Registry (MLflow)
- Backup в S3 с версионированием
- Возможность rollback к предыдущей версии за 5 минут

**Data:**
- Daily backups базы данных (PostgreSQL)
- Point-in-time recovery capability
- Retention: 30 дней

**Configuration:**
- GitOps подход: все конфиги в Git
- Infrastructure as Code (Terraform)

### 7.3 Incident Response

**Runbook для типичных инцидентов:**
1. API не отвечает → проверить health checks, restart pods
2. Высокий latency → check load, scale up, investigate slow queries
3. Деградация метрик → проверить drift, trigger retraining
4. Data pipeline failure → check logs, retry failed jobs

**Escalation path:**
- L1: On-call engineer (реагирование в течение 15 минут)
- L2: Team lead (если проблема не решена за 30 минут)
- L3: Architect / CTO (критичные инциденты)

---

## Заключение

Представленная архитектура антифрод-системы:

✅ **Модульная** - легко развивать и поддерживать  
✅ **Масштабируемая** - справляется с нагрузкой до 400 транзакций/сек  
✅ **Отказоустойчивая** - SLA 99.9%, быстрое восстановление  
✅ **Наблюдаемая** - полный мониторинг и алертинг  
✅ **Безопасная** - защита на всех уровнях  
✅ **Адаптивная** - автоматическое реагирование на drift

Такая архитектура позволит успешно реализовать проект в установленные сроки и достичь поставленных бизнес-целей.
