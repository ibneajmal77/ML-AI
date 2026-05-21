# PRODUCTION DEPLOYMENT SPEC: Enterprise RAG Knowledge Assistant
### Companion to PROJECT_SPEC.md — Read that first

> **Purpose:** Everything needed to take the built application from a local
> `docker-compose up` to a production deployment on Azure — with zero-downtime deploys,
> autoscaling, secrets management, and a full CI/CD pipeline.
>
> Two deployment options are covered:
> - **Option A — Azure Container Apps (ACA):** Recommended. Simpler, serverless-style scaling,
>   less ops overhead. Right for this project at initial production scale.
> - **Option B — Azure Kubernetes Service (AKS):** For higher scale (>500K chunks, complex routing,
>   multiple tenants with dedicated pods). More control, more ops work.

---

## 1. WHAT CHANGES FROM LOCAL TO PRODUCTION

In `docker-compose.yml` you run everything in Docker on one machine.
In production, each self-managed container gets replaced with a managed Azure service:

```
LOCAL (docker-compose)           PRODUCTION (Azure managed)
──────────────────────────────   ──────────────────────────────────────────
postgres container           →   Azure Database for PostgreSQL Flexible Server
                                  (pgvector extension enabled, private endpoint)

redis container              →   Azure Cache for Redis (Standard C1 tier)
                                  (private endpoint, no public access)

api container                →   Azure Container App (or AKS Deployment)
                                  (auto-scales 1–10 replicas based on HTTP RPS)

worker container             →   Azure Container App (or AKS Deployment)
                                  (auto-scales 0–5 replicas based on queue depth)

beat container               →   Azure Container App (1 replica, always-on)
                                  (never scale to zero — it drives schedules)

.env file                    →   Azure Key Vault (secrets)
                                  (Container Apps reference KV secrets directly)
```

---

## 2. COMPLETE PRODUCTION INFRASTRUCTURE

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Internet                                                               │
│    │                                                                    │
│    ▼                                                                    │
│  Azure Front Door + WAF (DDoS protection, SSL termination, geo routing) │
│    │                                                                    │
│    ▼                                                                    │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  Azure Virtual Network (VNet: 10.0.0.0/16)                       │  │
│  │                                                                   │  │
│  │  Subnet: apps (10.0.1.0/24)                                      │  │
│  │  ┌─────────────────────────────────────────┐                     │  │
│  │  │  Container Apps Environment              │                     │  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌───────┐  │                     │  │
│  │  │  │  api app │  │  worker  │  │  beat │  │                     │  │
│  │  │  │ (1–10)   │  │  (0–5)   │  │  (1)  │  │                     │  │
│  │  │  └──────────┘  └──────────┘  └───────┘  │                     │  │
│  │  └─────────────────────────────────────────┘                     │  │
│  │                                                                   │  │
│  │  Subnet: data (10.0.2.0/24)                                       │  │
│  │  ┌─────────────────────────────────────────┐                     │  │
│  │  │  Private Endpoints (no public access)    │                     │  │
│  │  │  ├── PostgreSQL Flexible Server          │                     │  │
│  │  │  ├── Azure Cache for Redis               │                     │  │
│  │  │  └── Azure Key Vault                     │                     │  │
│  │  └─────────────────────────────────────────┘                     │  │
│  └──────────────────────────────────────────────────────────────────┘  │
│                                                                         │
│  Supporting services (not in VNet):                                     │
│  ├── Azure Container Registry (ACR)     — stores Docker images          │
│  ├── Azure OpenAI                       — embeddings + GPT-4o           │
│  ├── Cohere API (external)              — reranking                     │
│  └── Langfuse (external / self-hosted)  — tracing                      │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 3. DIRECTORY ADDITIONS FOR PRODUCTION

Add these to the project structure from PROJECT_SPEC.md:

```
P3-enterprise-rag-knowledge-assistant/
│
├── infra/                              ← Infrastructure as Code (Bicep)
│   ├── main.bicep                      ← Entry point: deploys everything
│   ├── modules/
│   │   ├── registry.bicep              ← Azure Container Registry
│   │   ├── keyvault.bicep              ← Azure Key Vault + secrets
│   │   ├── network.bicep               ← VNet + subnets + private DNS zones
│   │   ├── postgres.bicep              ← PostgreSQL Flexible Server
│   │   ├── redis.bicep                 ← Azure Cache for Redis
│   │   └── container-apps.bicep        ← Container Apps Environment + 3 apps
│   └── parameters/
│       ├── dev.bicepparam              ← Dev environment values
│       └── prod.bicepparam             ← Prod environment values
│
├── k8s/                                ← Option B: Kubernetes manifests (AKS)
│   ├── namespace.yaml
│   ├── secrets.yaml                    ← External Secrets Operator config
│   ├── configmap.yaml
│   ├── deployments/
│   │   ├── api.yaml
│   │   ├── worker.yaml
│   │   └── beat.yaml
│   ├── services/
│   │   ├── api-service.yaml
│   │   └── worker-service.yaml
│   ├── hpa/
│   │   ├── api-hpa.yaml
│   │   └── worker-hpa.yaml
│   ├── ingress/
│   │   └── ingress.yaml
│   └── pdb/
│       └── api-pdb.yaml
│
├── Dockerfile                          ← Dev Dockerfile (already in PROJECT_SPEC)
├── Dockerfile.prod                     ← Hardened production Dockerfile
│
└── .github/
    └── workflows/
        ├── ci.yml                      ← Already in PROJECT_SPEC
        ├── ragas-eval-gate.yml         ← Already in PROJECT_SPEC
        ├── build-push.yml              ← Build + push image to ACR on merge to main
        └── deploy.yml                  ← Deploy new image to ACA or AKS
```

---

## 4. PRODUCTION DOCKERFILE (Dockerfile.prod)

The dev Dockerfile runs as root and installs dev tools. Production version is hardened.

```dockerfile
# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

# Install build deps only in builder stage (don't carry them to final image)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml .
# Install to a prefix directory so we can copy it cleanly
RUN pip install --no-cache-dir --prefix=/install -e .

# ── Stage 2: runtime ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Only the system libs needed at runtime (not build tools)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libmagic1 poppler-utils tesseract-ocr \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Non-root user — never run as root in production
RUN useradd --create-home --shell /bin/bash appuser
USER appuser

COPY --chown=appuser:appuser src/ src/
COPY --chown=appuser:appuser migrations/ migrations/

# Expose port
EXPOSE 8000

# Readiness: start with a short delay so the pool warms up before health checks
ENTRYPOINT ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000", \
            "--workers", "2", "--loop", "uvloop", "--http", "httptools"]
```

### Why this differs from dev Dockerfile

| Thing | Dev | Production |
|-------|-----|------------|
| Multi-stage build | No (single stage) | Yes — builder + runtime, keeps image small |
| Run as root | Yes | No — `appuser` non-root user |
| Build tools (gcc etc.) | Left in | Removed — only in builder stage |
| Uvicorn workers | 1 | 2 per container (tune based on CPU cores: 2 × CPUs) |
| uvloop + httptools | No | Yes — faster event loop and HTTP parser |

---

## 5. AZURE CONTAINER REGISTRY (ACR)

ACR stores the Docker images. GitHub Actions pushes here on merge to main.

### Bicep: infra/modules/registry.bicep

```bicep
param location string = resourceGroup().location
param registryName string  // e.g. "ragkbregistry"
param sku string = 'Basic'  // Basic for dev, Standard for prod (geo-replication)

resource acr 'Microsoft.ContainerRegistry/registries@2023-07-01' = {
  name: registryName
  location: location
  sku: {
    name: sku
  }
  properties: {
    adminUserEnabled: false  // Use managed identity, never admin credentials
    publicNetworkAccess: 'Enabled'  // ACR must be reachable from GitHub Actions runner
    zoneRedundancy: 'Disabled'      // Enable for prod: 'Enabled' (Standard+ SKU only)
  }
}

output loginServer string = acr.properties.loginServer
output acrId string = acr.id
```

### Naming convention for images

```
{acr-login-server}/rag-api:{git-sha}
{acr-login-server}/rag-worker:{git-sha}

Example:
  ragkbregistry.azurecr.io/rag-api:a3f9c12
  ragkbregistry.azurecr.io/rag-worker:a3f9c12
```

Same image is used for api, worker, and beat — the command is different, not the image.

---

## 6. AZURE KEY VAULT — SECRETS MANAGEMENT

No secrets in environment variables on the server. All secrets live in Key Vault.
Container Apps natively reference Key Vault secrets — no code changes needed.

### Bicep: infra/modules/keyvault.bicep

```bicep
param location string = resourceGroup().location
param keyVaultName string      // e.g. "rag-kb-kv-prod"
param containerAppPrincipalId string   // managed identity of the Container App

resource kv 'Microsoft.KeyVault/vaults@2023-07-01' = {
  name: keyVaultName
  location: location
  properties: {
    sku: {
      family: 'A'
      name: 'standard'
    }
    tenantId: subscription().tenantId
    enableRbacAuthorization: true   // use RBAC not access policies
    enableSoftDelete: true
    softDeleteRetentionInDays: 30
    publicNetworkAccess: 'Disabled' // only accessible from VNet private endpoint
    networkAcls: {
      defaultAction: 'Deny'
      bypass: 'AzureServices'
    }
  }
}

// Grant Container App identity permission to read secrets
resource kvSecretUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: kv
  name: guid(kv.id, containerAppPrincipalId, 'Key Vault Secrets User')
  properties: {
    roleDefinitionId: subscriptionResourceId('Microsoft.Authorization/roleDefinitions', '4633458b-17de-408a-b874-0445c86b69e6')
    principalId: containerAppPrincipalId
    principalType: 'ServicePrincipal'
  }
}

output keyVaultUri string = kv.properties.vaultUri
```

### Secrets to store in Key Vault

Create each as a Key Vault secret with these exact names:

```
azure-openai-api-key
azure-openai-endpoint
cohere-api-key
langfuse-secret-key
langfuse-public-key
postgres-connection-string          ← full connection URL with password
redis-connection-string             ← full connection URL with password
azure-di-key
azure-di-endpoint
```

### How Container Apps reference Key Vault secrets

In the Container App Bicep definition, secrets are declared like this:
```bicep
secrets: [
  {
    name: 'azure-openai-api-key'
    keyVaultUrl: '${kvUri}secrets/azure-openai-api-key'
    identity: containerAppManagedIdentity.id
  }
]
```
Then referenced in environment variables:
```bicep
env: [
  {
    name: 'AZURE_OPENAI_API_KEY'
    secretRef: 'azure-openai-api-key'  // matches secrets[].name above
  }
]
```

---

## 7. MANAGED POSTGRESQL (infra/modules/postgres.bicep)

Replace the postgres Docker container with Azure Database for PostgreSQL Flexible Server.
Must enable the `pgvector` extension — this requires explicit configuration.

```bicep
param location string = resourceGroup().location
param serverName string          // e.g. "rag-kb-postgres-prod"
param adminLogin string = 'ragadmin'
param adminPassword string       // passed from Key Vault reference
param subnetId string            // the data subnet ID for private access
param privateDnsZoneId string    // private DNS zone for postgres

resource postgresServer 'Microsoft.DBforPostgreSQL/flexibleServers@2023-06-01-preview' = {
  name: serverName
  location: location
  sku: {
    name: 'Standard_D2ds_v5'   // 2 vCores, 8 GB RAM — right for initial prod
    tier: 'GeneralPurpose'      // Required for pgvector
    // Scale up: Standard_D4ds_v5 (4 vCore) when query latency increases
  }
  properties: {
    administratorLogin: adminLogin
    administratorLoginPassword: adminPassword
    version: '16'
    storage: {
      storageSizeGB: 128
      autoGrow: 'Enabled'       // Disk expands automatically — no downtime
    }
    backup: {
      backupRetentionDays: 7
      geoRedundantBackup: 'Enabled'  // Point-in-time restore in secondary region
    }
    highAvailability: {
      mode: 'ZoneRedundant'    // Standby replica in different availability zone
      // Automatic failover in ~60s on primary failure
    }
    network: {
      delegatedSubnetResourceId: subnetId
      privateDnsZoneArmResourceId: privateDnsZoneId
    }
  }
}

// Enable pgvector — MUST be done explicitly or CREATE EXTENSION will fail
resource pgvectorExtension 'Microsoft.DBforPostgreSQL/flexibleServers/configurations@2023-06-01-preview' = {
  parent: postgresServer
  name: 'azure.extensions'
  properties: {
    value: 'VECTOR'   // Enables pgvector extension to be created by users
    source: 'user-override'
  }
}

// The database itself
resource database 'Microsoft.DBforPostgreSQL/flexibleServers/databases@2023-06-01-preview' = {
  parent: postgresServer
  name: 'ragdb'
  properties: {
    charset: 'UTF8'
    collation: 'en_US.utf8'
  }
}

output postgresServerFqdn string = postgresServer.properties.fullyQualifiedDomainName
```

### Connection string format

```
postgresql://ragadmin:{password}@{server-fqdn}:5432/ragdb?sslmode=require
```

`sslmode=require` is mandatory — Azure PostgreSQL rejects unencrypted connections.

### HNSW index tuning for production

After schema migration runs, set the search parameter for better recall:
```sql
SET hnsw.ef_search = 100;  -- Default is 40. Higher = better recall, slightly slower.
```
Set this at connection level via asyncpg: `await conn.execute("SET hnsw.ef_search = 100")` in the pool `init` callback.

---

## 8. MANAGED REDIS (infra/modules/redis.bicep)

```bicep
param location string = resourceGroup().location
param cacheName string       // e.g. "rag-kb-redis-prod"
param subnetId string

resource redisCache 'Microsoft.Cache/redis@2023-08-01' = {
  name: cacheName
  location: location
  properties: {
    sku: {
      name: 'Standard'   // Standard C1 = 1GB, replicated, ~60K ops/s
      family: 'C'
      capacity: 1
    }
    enableNonSslPort: false     // Force TLS — never allow unencrypted
    minimumTlsVersion: '1.2'
    redisConfiguration: {
      maxmemory-policy: 'allkeys-lru'  // Evict least-recently-used when full
      // This means old cached query results get evicted first
    }
  }
}

// Private endpoint — Redis not accessible from internet
resource privateEndpoint 'Microsoft.Network/privateEndpoints@2023-04-01' = {
  name: '${cacheName}-pe'
  location: location
  properties: {
    subnet: {
      id: subnetId
    }
    privateLinkServiceConnections: [
      {
        name: 'redis-connection'
        properties: {
          privateLinkServiceId: redisCache.id
          groupIds: ['redisCache']
        }
      }
    ]
  }
}

output redisHostName string = redisCache.properties.hostName
```

### Redis connection string format

```
rediss://:${accessKey}@${hostName}:6380/0
```

Note: `rediss://` (with double s) = TLS. Port `6380` = TLS port (not 6379).

---

## 9. OPTION A: AZURE CONTAINER APPS (infra/modules/container-apps.bicep)

This is the recommended option. ACA handles:
- Auto-scaling based on HTTP requests per second and queue depth (KEDA built-in)
- Ingress + TLS termination (no Nginx needed)
- Secret injection from Key Vault
- Rolling deployments (zero-downtime by default)
- Dapr sidecar available if needed later

### Container Apps Environment

The environment is the shared networking layer. All three apps (api, worker, beat) live in it.

```bicep
param location string = resourceGroup().location
param environmentName string      // e.g. "rag-kb-env-prod"
param infrastructureSubnetId string
param logAnalyticsWorkspaceId string
param logAnalyticsKey string

resource caEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: environmentName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalyticsWorkspaceId
        sharedKey: logAnalyticsKey
      }
    }
    vnetConfiguration: {
      infrastructureSubnetId: infrastructureSubnetId
      internal: false   // false = external ingress allowed. true = private only.
    }
    workloadProfiles: [
      {
        name: 'Consumption'    // Consumption = serverless, scale-to-zero
        workloadProfileType: 'Consumption'
      }
    ]
  }
}
```

### API Container App

```bicep
resource apiApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'rag-api'
  location: location
  identity: {
    type: 'SystemAssigned'   // Managed identity for Key Vault access
  }
  properties: {
    managedEnvironmentId: caEnvironment.id
    configuration: {
      ingress: {
        external: true       // Reachable from internet
        targetPort: 8000
        transport: 'http'
        traffic: [
          {
            weight: 100
            latestRevision: true
          }
        ]
      }
      secrets: [
        { name: 'azure-openai-api-key',    keyVaultUrl: '${kvUri}secrets/azure-openai-api-key',    identity: 'system' }
        { name: 'cohere-api-key',          keyVaultUrl: '${kvUri}secrets/cohere-api-key',          identity: 'system' }
        { name: 'postgres-connection-string', keyVaultUrl: '${kvUri}secrets/postgres-connection-string', identity: 'system' }
        { name: 'redis-connection-string', keyVaultUrl: '${kvUri}secrets/redis-connection-string', identity: 'system' }
        { name: 'langfuse-secret-key',     keyVaultUrl: '${kvUri}secrets/langfuse-secret-key',     identity: 'system' }
        { name: 'langfuse-public-key',     keyVaultUrl: '${kvUri}secrets/langfuse-public-key',     identity: 'system' }
      ]
    }
    template: {
      containers: [
        {
          name: 'rag-api'
          image: '${acrLoginServer}/rag-api:${imageTag}'
          resources: {
            cpu: json('1.0')    // 1 vCPU per replica
            memory: '2Gi'       // 2 GB RAM per replica
          }
          env: [
            { name: 'APP_ENV',                       value: 'production' }
            { name: 'LOG_LEVEL',                     value: 'INFO' }
            { name: 'AZURE_OPENAI_ENDPOINT',         value: azureOpenAIEndpoint }
            { name: 'AZURE_OPENAI_DEPLOYMENT',       value: 'gpt-4o' }
            { name: 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT', value: 'text-embedding-3-large' }
            { name: 'AZURE_OPENAI_API_VERSION',      value: '2024-10-21' }
            { name: 'LANGFUSE_HOST',                 value: 'https://cloud.langfuse.com' }
            { name: 'EMBEDDING_MODEL',               value: 'text-embedding-3-large' }
            { name: 'EMBEDDING_DIMS',                value: '3072' }
            { name: 'RERANKER_TOP_K',                value: '20' }
            { name: 'FINAL_TOP_K',                   value: '5' }
            { name: 'CHUNK_SIZE',                    value: '512' }
            { name: 'PARENT_CHUNK_SIZE',             value: '2048' }
            { name: 'CHUNK_OVERLAP',                 value: '50' }
            { name: 'MAX_CONTEXT_TOKENS',            value: '2000' }
            { name: 'CACHE_TTL_SECONDS',             value: '3600' }
            // Secrets referenced by name
            { name: 'AZURE_OPENAI_API_KEY',          secretRef: 'azure-openai-api-key' }
            { name: 'COHERE_API_KEY',                secretRef: 'cohere-api-key' }
            { name: 'DATABASE_URL',                  secretRef: 'postgres-connection-string' }
            { name: 'REDIS_URL',                     secretRef: 'redis-connection-string' }
            { name: 'LANGFUSE_SECRET_KEY',           secretRef: 'langfuse-secret-key' }
            { name: 'LANGFUSE_PUBLIC_KEY',           secretRef: 'langfuse-public-key' }
          ]
          probes: [
            {
              type: 'Liveness'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 10
              periodSeconds: 15
              failureThreshold: 3
            }
            {
              type: 'Readiness'
              httpGet: { path: '/health/ready', port: 8000 }
              initialDelaySeconds: 15
              periodSeconds: 10
              failureThreshold: 3
              // Checks DB + Redis. Pod removed from traffic until both are reachable.
            }
            {
              type: 'Startup'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 5
              periodSeconds: 5
              failureThreshold: 12   // 60s total startup window (BM25 seed time)
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1       // Never scale to zero (avoids cold start on query)
        maxReplicas: 10
        rules: [
          {
            name: 'http-scaling'
            http: {
              metadata: {
                concurrentRequests: '20'  // Add replica when >20 concurrent requests per replica
              }
            }
          }
        ]
      }
    }
  }
}
```

### Worker Container App

Same image as API, different command and scaling rule.

```bicep
resource workerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'rag-worker'
  // ... same identity + secrets as API app
  properties: {
    // No ingress — workers don't accept HTTP traffic
    template: {
      containers: [
        {
          name: 'rag-worker'
          image: '${acrLoginServer}/rag-api:${imageTag}'  // same image
          command: ['celery']
          args: ['-A', 'src.indexing.worker', 'worker',
                 '--loglevel=info', '-Q', 'ingest,reindex',
                 '--concurrency=4']
          resources: {
            cpu: json('2.0')   // Workers are CPU-heavy (embedding computation)
            memory: '4Gi'
          }
          // ... same env vars
        }
      ]
      scale: {
        minReplicas: 0   // Workers CAN scale to zero when queue is empty
        maxReplicas: 5
        rules: [
          {
            name: 'redis-queue-scaling'
            custom: {
              type: 'redis'
              metadata: {
                listName: 'celery'        // Celery's default queue list in Redis
                listLength: '5'           // Add worker when >5 items in queue
                address: redisHost
              }
              auth: [{ secretRef: 'redis-connection-string', triggerParameter: 'address' }]
            }
          }
        ]
      }
    }
  }
}
```

### Beat Container App

Celery beat runs periodic tasks. Must never scale to zero (would miss scheduled runs).
Must never have more than 1 replica (duplicate schedules = duplicate full re-indexes).

```bicep
resource beatApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'rag-beat'
  properties: {
    template: {
      containers: [
        {
          name: 'rag-beat'
          image: '${acrLoginServer}/rag-api:${imageTag}'
          command: ['celery']
          args: ['-A', 'src.indexing.worker', 'beat', '--loglevel=info']
          resources: {
            cpu: json('0.25')   // Beat is very lightweight
            memory: '512Mi'
          }
        }
      ]
      scale: {
        minReplicas: 1   // Always running
        maxReplicas: 1   // NEVER more than 1 — prevents duplicate scheduled jobs
      }
    }
  }
}
```

---

## 10. OPTION B: KUBERNETES (AKS) — k8s/ directory

Use this if you need more control: custom node pools, GPU nodes for local embedding,
namespace isolation per tenant, complex network policies, or scale beyond 10 replicas.

### k8s/namespace.yaml

```yaml
apiVersion: v1
kind: Namespace
metadata:
  name: rag-system
  labels:
    app: enterprise-rag
```

### k8s/configmap.yaml

Non-secret config values. Secrets come from Azure Key Vault via External Secrets Operator.

```yaml
apiVersion: v1
kind: ConfigMap
metadata:
  name: rag-config
  namespace: rag-system
data:
  APP_ENV: "production"
  LOG_LEVEL: "INFO"
  AZURE_OPENAI_DEPLOYMENT: "gpt-4o"
  AZURE_OPENAI_EMBEDDING_DEPLOYMENT: "text-embedding-3-large"
  AZURE_OPENAI_API_VERSION: "2024-10-21"
  LANGFUSE_HOST: "https://cloud.langfuse.com"
  EMBEDDING_MODEL: "text-embedding-3-large"
  EMBEDDING_DIMS: "3072"
  RERANKER_TOP_K: "20"
  FINAL_TOP_K: "5"
  CHUNK_SIZE: "512"
  PARENT_CHUNK_SIZE: "2048"
  CHUNK_OVERLAP: "50"
  MAX_CONTEXT_TOKENS: "2000"
  CACHE_TTL_SECONDS: "3600"
  RRF_K: "60"
  CELERY_BROKER_URL: "rediss://:$(REDIS_KEY)@$(REDIS_HOST):6380/1"
  CELERY_RESULT_BACKEND: "rediss://:$(REDIS_KEY)@$(REDIS_HOST):6380/2"
```

### k8s/secrets.yaml

Use External Secrets Operator (ESO) to sync from Azure Key Vault into K8s Secrets.
Never commit actual secret values to git.

```yaml
apiVersion: external-secrets.io/v1beta1
kind: ExternalSecret
metadata:
  name: rag-secrets
  namespace: rag-system
spec:
  refreshInterval: 5m              # Sync every 5 minutes
  secretStoreRef:
    name: azure-keyvault-store
    kind: ClusterSecretStore
  target:
    name: rag-secrets              # Creates a K8s Secret named "rag-secrets"
    creationPolicy: Owner
  data:
    - secretKey: AZURE_OPENAI_API_KEY
      remoteRef:
        key: azure-openai-api-key
    - secretKey: COHERE_API_KEY
      remoteRef:
        key: cohere-api-key
    - secretKey: DATABASE_URL
      remoteRef:
        key: postgres-connection-string
    - secretKey: REDIS_URL
      remoteRef:
        key: redis-connection-string
    - secretKey: LANGFUSE_SECRET_KEY
      remoteRef:
        key: langfuse-secret-key
    - secretKey: LANGFUSE_PUBLIC_KEY
      remoteRef:
        key: langfuse-public-key
    - secretKey: AZURE_OPENAI_ENDPOINT
      remoteRef:
        key: azure-openai-endpoint
```

### k8s/deployments/api.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-api
  namespace: rag-system
  labels:
    app: rag-api
    version: "1.0.0"
spec:
  replicas: 2                      # Start with 2; HPA will scale from here
  selector:
    matchLabels:
      app: rag-api
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 0            # Zero-downtime: never take a pod down before new one is ready
      maxSurge: 1                  # Spin up 1 extra pod during deploy
  template:
    metadata:
      labels:
        app: rag-api
    spec:
      serviceAccountName: rag-api-sa   # Workload identity for Key Vault access

      # Spread replicas across availability zones
      topologySpreadConstraints:
        - maxSkew: 1
          topologyKey: topology.kubernetes.io/zone
          whenUnsatisfiable: DoNotSchedule
          labelSelector:
            matchLabels:
              app: rag-api

      containers:
        - name: rag-api
          image: ragkbregistry.azurecr.io/rag-api:IMAGE_TAG
          ports:
            - containerPort: 8000

          resources:
            requests:
              cpu: "500m"          # 0.5 vCPU guaranteed
              memory: "1Gi"        # 1 GB guaranteed
            limits:
              cpu: "2000m"         # Burst up to 2 vCPU
              memory: "2Gi"        # Never exceed 2 GB

          envFrom:
            - configMapRef:
                name: rag-config
            - secretRef:
                name: rag-secrets

          livenessProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 10
            periodSeconds: 15
            failureThreshold: 3

          readinessProbe:
            httpGet:
              path: /health/ready
              port: 8000
            initialDelaySeconds: 15
            periodSeconds: 10
            failureThreshold: 3
            # Pod is removed from Service endpoints until DB + Redis are both reachable

          startupProbe:
            httpGet:
              path: /health
              port: 8000
            initialDelaySeconds: 5
            periodSeconds: 5
            failureThreshold: 12   # 60 second startup window (BM25 seed from DB)

          lifecycle:
            preStop:
              exec:
                command: ["/bin/sh", "-c", "sleep 5"]
                # 5s grace period so load balancer routes traffic away before pod stops

      terminationGracePeriodSeconds: 30
```

### k8s/deployments/worker.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-worker
  namespace: rag-system
spec:
  replicas: 1
  selector:
    matchLabels:
      app: rag-worker
  strategy:
    type: RollingUpdate
    rollingUpdate:
      maxUnavailable: 1   # OK to have 0 workers briefly (tasks are queued in Redis)
      maxSurge: 1
  template:
    metadata:
      labels:
        app: rag-worker
    spec:
      containers:
        - name: rag-worker
          image: ragkbregistry.azurecr.io/rag-api:IMAGE_TAG
          command: ["celery"]
          args:
            - "-A"
            - "src.indexing.worker"
            - "worker"
            - "--loglevel=info"
            - "-Q"
            - "ingest,reindex"
            - "--concurrency=4"
          resources:
            requests:
              cpu: "1000m"
              memory: "2Gi"
            limits:
              cpu: "4000m"    # Workers can burst for embedding computation
              memory: "4Gi"
          envFrom:
            - configMapRef:
                name: rag-config
            - secretRef:
                name: rag-secrets
```

### k8s/deployments/beat.yaml

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: rag-beat
  namespace: rag-system
spec:
  replicas: 1             # ALWAYS exactly 1 — beat must be a singleton
  selector:
    matchLabels:
      app: rag-beat
  strategy:
    type: Recreate        # NOT RollingUpdate — must stop old before starting new
                          # Two beat processes = double-scheduled jobs
  template:
    metadata:
      labels:
        app: rag-beat
    spec:
      containers:
        - name: rag-beat
          image: ragkbregistry.azurecr.io/rag-api:IMAGE_TAG
          command: ["celery"]
          args: ["-A", "src.indexing.worker", "beat", "--loglevel=info"]
          resources:
            requests:
              cpu: "100m"
              memory: "256Mi"
            limits:
              cpu: "500m"
              memory: "512Mi"
          envFrom:
            - configMapRef:
                name: rag-config
            - secretRef:
                name: rag-secrets
```

### k8s/services/api-service.yaml

```yaml
apiVersion: v1
kind: Service
metadata:
  name: rag-api-service
  namespace: rag-system
spec:
  selector:
    app: rag-api
  ports:
    - port: 80
      targetPort: 8000
  type: ClusterIP    # Internal only — traffic comes in through Ingress
```

### k8s/hpa/api-hpa.yaml

Horizontal Pod Autoscaler — scales API based on CPU.

```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: rag-api-hpa
  namespace: rag-system
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: rag-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
    - type: Resource
      resource:
        name: cpu
        target:
          type: Utilization
          averageUtilization: 60    # Scale up when avg CPU > 60%
    - type: Resource
      resource:
        name: memory
        target:
          type: Utilization
          averageUtilization: 75    # Scale up when avg memory > 75%
  behavior:
    scaleUp:
      stabilizationWindowSeconds: 60    # Wait 60s before scaling up again
      policies:
        - type: Pods
          value: 2
          periodSeconds: 60             # Add max 2 pods per 60 seconds
    scaleDown:
      stabilizationWindowSeconds: 300   # Wait 5 minutes before scaling down
      policies:
        - type: Pods
          value: 1
          periodSeconds: 120            # Remove max 1 pod per 2 minutes
```

### k8s/hpa/worker-hpa.yaml

Workers scale based on Redis queue depth (KEDA).

```yaml
apiVersion: keda.sh/v1alpha1
kind: ScaledObject
metadata:
  name: rag-worker-scaledobject
  namespace: rag-system
spec:
  scaleTargetRef:
    name: rag-worker
  minReplicaCount: 0        # Scale to zero when queue is empty
  maxReplicaCount: 5
  triggers:
    - type: redis
      metadata:
        address: rag-redis-host:6380
        listName: celery        # Celery default queue list
        listLength: "5"         # 1 worker handles up to 5 queued tasks
        enableTLS: "true"
      authenticationRef:
        name: redis-trigger-auth
```

### k8s/ingress/ingress.yaml

```yaml
apiVersion: networking.k8s.io/v1
kind: Ingress
metadata:
  name: rag-ingress
  namespace: rag-system
  annotations:
    kubernetes.io/ingress.class: nginx
    nginx.ingress.kubernetes.io/ssl-redirect: "true"
    nginx.ingress.kubernetes.io/proxy-body-size: "50m"     # Allow large document uploads
    nginx.ingress.kubernetes.io/proxy-read-timeout: "60"   # 60s for slow reranker queries
    # Rate limiting: 30 requests per minute per IP
    nginx.ingress.kubernetes.io/limit-rps: "30"
    nginx.ingress.kubernetes.io/limit-connections: "10"
    cert-manager.io/cluster-issuer: letsencrypt-prod
spec:
  tls:
    - hosts:
        - rag-api.yourcompany.com
      secretName: rag-tls-cert
  rules:
    - host: rag-api.yourcompany.com
      http:
        paths:
          - path: /
            pathType: Prefix
            backend:
              service:
                name: rag-api-service
                port:
                  number: 80
```

### k8s/pdb/api-pdb.yaml

Pod Disruption Budget — ensures at least 1 API pod is always running during node upgrades.

```yaml
apiVersion: policy/v1
kind: PodDisruptionBudget
metadata:
  name: rag-api-pdb
  namespace: rag-system
spec:
  minAvailable: 1      # Always keep at least 1 pod running
  selector:
    matchLabels:
      app: rag-api
```

---

## 11. DATABASE MIGRATION STRATEGY (PRODUCTION)

Never run `docker-entrypoint-initdb.d` in production (that's for dev only).
Production migrations run as a Kubernetes Job before the Deployment rolls out.

### Migration Job pattern

Run this job in CI before deploying the new container version:

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: rag-db-migrate-v1
  namespace: rag-system
spec:
  backoffLimit: 3
  template:
    spec:
      restartPolicy: OnFailure
      containers:
        - name: migrate
          image: ragkbregistry.azurecr.io/rag-api:IMAGE_TAG
          command: ["psql"]
          args:
            - "$(DATABASE_URL)"
            - "-f"
            - "migrations/001_initial_schema.sql"
          envFrom:
            - secretRef:
                name: rag-secrets
```

### Safe migration rules

Migrations on a live database must follow these rules to avoid downtime:

1. **Add columns as nullable first.** Never add NOT NULL without a DEFAULT on a live table.
   If you need NOT NULL: add nullable → backfill → add constraint.

2. **Add indexes CONCURRENTLY.** `CREATE INDEX CONCURRENTLY` builds without locking the table.
   Normal `CREATE INDEX` locks for minutes on large tables.
   ```sql
   CREATE INDEX CONCURRENTLY IF NOT EXISTS idx_chunks_new ON chunks(new_column);
   ```

3. **Never rename or drop columns in the same migration as the app deployment.**
   Deploy the new app first (still reads old column), then drop the old column in a
   follow-up migration. Two-phase: add → deploy → remove.

4. **Test against production data size.** Run `EXPLAIN ANALYZE` on new queries.
   An index that helps on 1K rows may not help on 500K rows.

---

## 12. CI/CD DEPLOYMENT PIPELINES

### .github/workflows/build-push.yml — runs on merge to main

```yaml
name: Build and Push to ACR

on:
  push:
    branches: [main]

jobs:
  build-push:
    runs-on: ubuntu-latest
    permissions:
      id-token: write    # Required for OIDC auth to Azure (no stored credentials)
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Azure login (OIDC — no secrets needed)
        uses: azure/login@v2
        with:
          client-id: ${{ secrets.AZURE_CLIENT_ID }}
          tenant-id: ${{ secrets.AZURE_TENANT_ID }}
          subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}

      - name: Log in to ACR
        run: az acr login --name ${{ vars.ACR_NAME }}

      - name: Build and push
        run: |
          IMAGE_TAG="${{ vars.ACR_LOGIN_SERVER }}/rag-api:${{ github.sha }}"
          docker build -f Dockerfile.prod -t $IMAGE_TAG .
          docker push $IMAGE_TAG
          # Also tag as latest
          docker tag $IMAGE_TAG "${{ vars.ACR_LOGIN_SERVER }}/rag-api:latest"
          docker push "${{ vars.ACR_LOGIN_SERVER }}/rag-api:latest"

      - name: Save image tag as output
        id: image
        run: echo "tag=${{ github.sha }}" >> $GITHUB_OUTPUT

    outputs:
      image_tag: ${{ steps.image.outputs.tag }}
```

### .github/workflows/deploy.yml — runs after build-push succeeds

```yaml
name: Deploy to Production

on:
  workflow_run:
    workflows: ["Build and Push to ACR"]
    types: [completed]
    branches: [main]

jobs:
  deploy:
    if: ${{ github.event.workflow_run.conclusion == 'success' }}
    runs-on: ubuntu-latest
    environment: production     # Requires manual approval in GitHub Environments settings
    permissions:
      id-token: write
      contents: read

    steps:
      - uses: actions/checkout@v4

      - name: Azure login
        uses: azure/login@v2
        with:
          client-id: ${{ secrets.AZURE_CLIENT_ID }}
          tenant-id: ${{ secrets.AZURE_TENANT_ID }}
          subscription-id: ${{ secrets.AZURE_SUBSCRIPTION_ID }}

      # ── Option A: Azure Container Apps deploy ─────────────────────────────
      - name: Deploy API to Container Apps
        run: |
          az containerapp update \
            --name rag-api \
            --resource-group ${{ vars.RESOURCE_GROUP }} \
            --image "${{ vars.ACR_LOGIN_SERVER }}/rag-api:${{ github.sha }}"

      - name: Deploy Worker to Container Apps
        run: |
          az containerapp update \
            --name rag-worker \
            --resource-group ${{ vars.RESOURCE_GROUP }} \
            --image "${{ vars.ACR_LOGIN_SERVER }}/rag-api:${{ github.sha }}"

      - name: Deploy Beat to Container Apps
        run: |
          az containerapp update \
            --name rag-beat \
            --resource-group ${{ vars.RESOURCE_GROUP }} \
            --image "${{ vars.ACR_LOGIN_SERVER }}/rag-api:${{ github.sha }}"

      # ── Option B: AKS deploy (use this block instead if using AKS) ───────
      # - name: Get AKS credentials
      #   run: az aks get-credentials --resource-group ${{ vars.RESOURCE_GROUP }} --name ${{ vars.AKS_CLUSTER }}
      #
      # - name: Run DB migration job
      #   run: |
      #     sed -i 's/IMAGE_TAG/${{ github.sha }}/g' k8s/jobs/migrate.yaml
      #     kubectl apply -f k8s/jobs/migrate.yaml
      #     kubectl wait --for=condition=complete job/rag-db-migrate --timeout=120s -n rag-system
      #
      # - name: Deploy new image
      #   run: |
      #     kubectl set image deployment/rag-api rag-api=${{ vars.ACR_LOGIN_SERVER }}/rag-api:${{ github.sha }} -n rag-system
      #     kubectl set image deployment/rag-worker rag-worker=${{ vars.ACR_LOGIN_SERVER }}/rag-api:${{ github.sha }} -n rag-system
      #     kubectl set image deployment/rag-beat rag-beat=${{ vars.ACR_LOGIN_SERVER }}/rag-api:${{ github.sha }} -n rag-system
      #     kubectl rollout status deployment/rag-api -n rag-system --timeout=300s

      - name: Verify deployment health
        run: |
          sleep 30   # Wait for readiness probes to pass
          curl -f https://rag-api.yourcompany.com/health/ready || exit 1
```

### Full pipeline flow on a PR merge

```
Push to main
    │
    ▼
[ci.yml] lint + unit tests                (2 min)
    │
    ▼
[ragas-eval-gate.yml] RAGAS evaluation    (5–10 min)
    │ blocks merge if faithfulness < 0.85
    ▼
[build-push.yml] docker build + push ACR  (3–5 min)
    │
    ▼
[deploy.yml] requires manual approval in GitHub    ← production gate
    │         (configured in GitHub → Settings → Environments → production)
    ▼
Run DB migration job (if any)
    │
    ▼
Deploy new image to ACA / AKS (rolling update)
    │
    ▼
Verify /health/ready returns 200
```

---

## 13. PRODUCTION ENVIRONMENT VARIABLES REFERENCE

Full list of what each service needs. All secrets come from Key Vault.

### API and Worker (same env vars)

| Variable | Source | Value in Production |
|---|---|---|
| `APP_ENV` | ConfigMap | `production` |
| `LOG_LEVEL` | ConfigMap | `INFO` |
| `AZURE_OPENAI_ENDPOINT` | Key Vault | `https://your-resource.openai.azure.com/` |
| `AZURE_OPENAI_API_KEY` | Key Vault | actual key |
| `AZURE_OPENAI_DEPLOYMENT` | ConfigMap | `gpt-4o` |
| `AZURE_OPENAI_EMBEDDING_DEPLOYMENT` | ConfigMap | `text-embedding-3-large` |
| `AZURE_OPENAI_API_VERSION` | ConfigMap | `2024-10-21` |
| `DATABASE_URL` | Key Vault | `postgresql://...@server:5432/ragdb?sslmode=require` |
| `REDIS_URL` | Key Vault | `rediss://:key@host:6380/0` |
| `CELERY_BROKER_URL` | Key Vault (derived) | `rediss://:key@host:6380/1` |
| `CELERY_RESULT_BACKEND` | Key Vault (derived) | `rediss://:key@host:6380/2` |
| `COHERE_API_KEY` | Key Vault | actual key |
| `LANGFUSE_PUBLIC_KEY` | Key Vault | `pk-lf-...` |
| `LANGFUSE_SECRET_KEY` | Key Vault | `sk-lf-...` |
| `LANGFUSE_HOST` | ConfigMap | `https://cloud.langfuse.com` |
| `EMBEDDING_MODEL` | ConfigMap | `text-embedding-3-large` |
| `EMBEDDING_DIMS` | ConfigMap | `3072` |
| `RERANKER_TOP_K` | ConfigMap | `20` |
| `FINAL_TOP_K` | ConfigMap | `5` |
| `CHUNK_SIZE` | ConfigMap | `512` |
| `PARENT_CHUNK_SIZE` | ConfigMap | `2048` |
| `CACHE_TTL_SECONDS` | ConfigMap | `3600` |

---

## 14. PRODUCTION HARDENING CHECKLIST

These are not optional. Each one addresses a real production failure mode.

### API Security

- [ ] **Authentication**: Add API key middleware before the query and ingest routes.
  Every request must include `X-API-Key: {key}` header. Validate against a set of
  hashed keys stored in Key Vault. Return 401 if missing/invalid.
  Do NOT add auth to `/health` and `/health/ready` (load balancer needs those unauthenticated).

- [ ] **Rate limiting**: Enforce per-IP and per-API-key limits.
  Use `slowapi` library: `Limiter(key_func=get_remote_address)`.
  Limits: `@limiter.limit("30/minute")` on `/v1/query`, `"10/minute"` on `/v1/ingest`.

- [ ] **Request size limit**: Ingest endpoint accepts file uploads. Cap at 50MB.
  In Uvicorn: `--limit-max-requests` or in FastAPI: `app.add_middleware(TrustedHostMiddleware, ...)`.
  In Nginx Ingress: `nginx.ingress.kubernetes.io/proxy-body-size: "50m"`.

- [ ] **Timeout enforcement**: Set `READ_TIMEOUT=30` and `WRITE_TIMEOUT=60` on Uvicorn.
  Prevents hung requests from blocking worker threads.

### Container Security

- [ ] **Non-root user**: Already in `Dockerfile.prod` (`USER appuser`).
- [ ] **Read-only filesystem**: Add `readOnlyRootFilesystem: true` in K8s securityContext.
  Mount `/tmp` as an emptyDir volume (needed for unstructured.io temp files).
- [ ] **Drop capabilities**: `securityContext.capabilities.drop: [ALL]`.
- [ ] **No privilege escalation**: `allowPrivilegeEscalation: false`.

```yaml
# Add to containers section in all Deployment YAMLs
securityContext:
  runAsNonRoot: true
  runAsUser: 1000
  readOnlyRootFilesystem: true
  allowPrivilegeEscalation: false
  capabilities:
    drop: [ALL]
volumeMounts:
  - name: tmp
    mountPath: /tmp
volumes:
  - name: tmp
    emptyDir: {}
```

### Database Security

- [ ] **Connection over SSL**: `sslmode=require` in DATABASE_URL.
- [ ] **Minimum privilege**: Create a dedicated DB user with only SELECT/INSERT/UPDATE/DELETE
  on the four tables. No CREATE TABLE, no DROP, no TRUNCATE.
  ```sql
  CREATE USER ragapp WITH PASSWORD '...';
  GRANT SELECT, INSERT, UPDATE, DELETE ON documents, parent_chunks, chunks, query_cache TO ragapp;
  GRANT USAGE ON SCHEMA public TO ragapp;
  ```
- [ ] **Connection pool limits**: `max_size=10` per pod. With 10 API replicas = 100 max connections.
  PostgreSQL default max_connections is 100. Set to 200 in Flexible Server config to be safe.

### Secrets

- [ ] **No secrets in git** — ever. `.env` is in `.gitignore`. All secrets in Key Vault.
- [ ] **Rotate secrets**: Azure Key Vault supports secret versioning. Rotate API keys quarterly.
  Container Apps / K8s picks up new version within `refreshInterval` (5 minutes).
- [ ] **Audit log**: Enable Key Vault diagnostic settings → Log Analytics. Every secret read is logged.

---

## 15. OBSERVABILITY IN PRODUCTION

### Logging

All application logs go to Azure Log Analytics (via Container Apps built-in or AKS + Azure Monitor).

Set structured JSON logging in production. In `src/config.py`, when `APP_ENV == "production"`:

```python
import logging
import json

class JSONFormatter(logging.Formatter):
    def format(self, record):
        return json.dumps({
            "timestamp": self.formatTime(record),
            "level":     record.levelname,
            "logger":    record.name,
            "message":   record.getMessage(),
            "trace_id":  getattr(record, "trace_id", None),
        })
```

Every log line is now queryable in Log Analytics with KQL:

```kql
ContainerAppConsoleLogs
| where ContainerName == "rag-api"
| where Log contains "faithfulness"
| project TimeGenerated, Log
| order by TimeGenerated desc
```

### Metrics (Azure Monitor + Grafana)

Four dashboards to create. Connect Grafana to Azure Monitor as data source.

**Dashboard 1 — Latency**
- P50, P95, P99 of `/v1/query` response time
- Broken down by: cache hit vs miss, with reranker vs without
- Alert: P95 > 3000ms sustained 5 min → PagerDuty

**Dashboard 2 — Quality**
- Rolling 1-hour faithfulness score (from RAGAS eval scheduled nightly)
- Answer relevancy score
- Alert: faithfulness < 0.85 → PagerDuty

**Dashboard 3 — Cost**
- Azure OpenAI token usage per day (prompt + completion tokens separate)
- Cohere API calls per day
- Cost per query (Azure OpenAI cost ÷ query count)

**Dashboard 4 — Operations**
- Cache hit rate (Redis hits ÷ total queries)
- Indexing queue depth (Celery queue length in Redis)
- Documents indexed per day
- BM25 index size (chunk count)

### Langfuse Traces

Every production query has a full trace visible at `cloud.langfuse.com`:

```
Trace: rag-query (total: 1240ms)
├── embedding     :  45ms  → input: {query: "What is...", use_hyde: false}
├── retrieval     : 120ms  → output: {candidates: 20}
├── rerank        : 380ms  → output: {reranked: 5}
├── assemble      :  15ms  → output: {tokens_used: 1840}
└── generate      : 680ms  → output: {answer_length: 312}
```

Use Langfuse to debug quality issues: open a failing query, see exactly which chunks
were retrieved, what got reranked to top-5, what context was assembled.

---

## 16. ROLLBACK PROCEDURE

### Azure Container Apps rollback

ACA keeps the previous revision. Rollback = redirect traffic to old revision.

```bash
# List revisions
az containerapp revision list --name rag-api --resource-group $RG

# Activate the previous revision
az containerapp revision activate \
  --name rag-api \
  --resource-group $RG \
  --revision rag-api--previous-revision-name

# Move 100% traffic to old revision
az containerapp ingress traffic set \
  --name rag-api \
  --resource-group $RG \
  --revision-weight rag-api--previous-revision-name=100
```

### AKS rollback

```bash
kubectl rollout undo deployment/rag-api -n rag-system
kubectl rollout status deployment/rag-api -n rag-system
```

### Database rollback

Write a corresponding `002_rollback_001.sql` for every migration.
If migration breaks something: `psql $DATABASE_URL -f migrations/002_rollback_001.sql`

Example rollback for 001:
```sql
-- Rollback for 001_initial_schema.sql
DROP TABLE IF EXISTS query_cache;
DROP TABLE IF EXISTS chunks;
DROP TABLE IF EXISTS parent_chunks;
DROP TABLE IF EXISTS documents;
DROP EXTENSION IF EXISTS vector;
```

---

## 17. SCALING GUIDE

When to scale what, and how:

### API replicas
- Scale trigger: P95 latency > 2s OR CPU avg > 60%
- Scale step: +2 replicas at a time
- Max: 10 replicas (constrained by DB connection pool: 10 replicas × 10 conns = 100)
- If need more: increase PostgreSQL max_connections and asyncpg pool max_size

### Worker replicas
- Scale trigger: Celery queue depth > 5 tasks (KEDA handles this automatically)
- Scale to zero when idle (no ingest traffic)
- Scale step: +1 replica per 5 queued tasks

### PostgreSQL
- Scale trigger: CPU > 70% sustained, or query latency > 500ms on vector search
- Scale action: Change SKU from D2ds_v5 → D4ds_v5 (Azure Portal, ~2 min, no downtime with HA)
- At 500K+ chunks: consider Qdrant for vector search (more ANN-optimized than pgvector at scale)

### Redis
- Scale trigger: Redis CPU > 70%, or eviction rate > 0 (cache full)
- Scale action: Upgrade C1 → C2 (2GB) in Azure Portal

### When to move from ACA to AKS
- Need GPU nodes for local embedding (sentence-transformers at scale)
- Need per-tenant namespace isolation with network policies
- Need more than 10 API replicas
- Need custom admission controllers or complex RBAC

---

## 18. COST ESTIMATES (Monthly, Production)

Approximate Azure costs for a medium-scale enterprise deployment (10K queries/day):

```
Azure Database for PostgreSQL Flexible Server (D2ds_v5, HA)  ~$200/month
Azure Cache for Redis (Standard C1, 1GB)                     ~$55/month
Azure Container Apps (api: avg 2 replicas, 1vCPU/2GB)        ~$120/month
Azure Container Apps (worker: avg 0.5 replicas)              ~$30/month
Azure Container Apps (beat: 1 replica, 0.25vCPU)             ~$15/month
Azure Container Registry (Standard, 100GB)                   ~$20/month
Azure Key Vault (Standard, <10K operations/month)            ~$5/month
Azure Log Analytics (5GB/day ingestion)                      ~$75/month
Azure Front Door (Standard, 10TB egress)                     ~$90/month
─────────────────────────────────────────────────────────────────────────
Infrastructure subtotal                                      ~$610/month

Azure OpenAI (10K queries × avg 3K prompt tokens + 200 output)  ~$180/month
Cohere Rerank (10K queries × 20 docs × 100 tokens each)         ~$100/month
─────────────────────────────────────────────────────────────────────────
AI API subtotal                                              ~$280/month

TOTAL ESTIMATE                                               ~$890/month
```

Cost optimization levers:
- Redis cache (38% hit rate) saves ~$38/month in Cohere calls
- Scale API to 1 replica during off-hours (nights/weekends) → saves ~$40/month
- Use `text-embedding-3-small` instead of `large` for cheaper embeddings (lower quality)
- Qdrant Cloud free tier for vector search if migrating off pgvector

---

## 19. FIRST DEPLOY CHECKLIST

Run through this in order on first production deploy:

```
Infrastructure
  [ ] Run: az deployment group create --template-file infra/main.bicep --parameters infra/parameters/prod.bicepparam
  [ ] Verify PostgreSQL is reachable from ACA environment (private endpoint DNS resolves)
  [ ] Verify Redis is reachable from ACA environment
  [ ] Verify all Key Vault secrets are populated (az keyvault secret list --vault-name rag-kb-kv-prod)

Database
  [ ] Connect to PostgreSQL via Azure Cloud Shell or bastion
  [ ] Run: psql $DATABASE_URL -f migrations/001_initial_schema.sql
  [ ] Verify: SELECT * FROM pg_extension WHERE extname = 'vector';  ← must return 1 row
  [ ] Verify tables exist: \dt in psql

Container Registry
  [ ] az acr login --name ragkbregistry
  [ ] docker build -f Dockerfile.prod -t ragkbregistry.azurecr.io/rag-api:v1.0.0 .
  [ ] docker push ragkbregistry.azurecr.io/rag-api:v1.0.0

Container Apps
  [ ] Deploy api, worker, beat apps
  [ ] az containerapp show --name rag-api --query "properties.latestRevisionFqdn"  ← get URL
  [ ] curl https://{fqdn}/health  ← must return {"status": "ok"}
  [ ] curl https://{fqdn}/health/ready  ← must return {"status": "ready", "db": "ok", "cache": "ok"}

Ingest Test
  [ ] curl -X POST https://{fqdn}/v1/ingest -F "file=@test.pdf" -F "source_url=..." -F "owner_dept=test"
  [ ] Watch worker logs: az containerapp logs show --name rag-worker
  [ ] Verify chunks in DB: SELECT COUNT(*) FROM chunks;

Query Test
  [ ] curl -X POST https://{fqdn}/v1/query -H "Content-Type: application/json" \
      -d '{"query": "test question", "user_clearance": 1}'
  [ ] Verify response has answer + citations + latency_ms
  [ ] Open Langfuse dashboard — verify trace appeared

Monitoring
  [ ] Grafana connected to Azure Monitor
  [ ] PagerDuty alert configured (faithfulness < 0.85)
  [ ] Log Analytics query for errors works

CI/CD
  [ ] GitHub Environments "production" configured with required reviewer
  [ ] AZURE_CLIENT_ID, AZURE_TENANT_ID, AZURE_SUBSCRIPTION_ID set as GitHub secrets
  [ ] ACR_NAME, ACR_LOGIN_SERVER, RESOURCE_GROUP set as GitHub variables
  [ ] Test full pipeline: merge a trivial change to main, watch all workflows pass
```
