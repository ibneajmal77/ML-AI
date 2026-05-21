param location string = resourceGroup().location
param environmentName string
param infrastructureSubnetId string
param acrLoginServer string
param postgresServerFqdn string
param redisHostName string
param kvUri string
param azureOpenAIEndpoint string
param imageTag string = 'latest'

resource logAnalytics 'Microsoft.OperationalInsights/workspaces@2022-10-01' = {
  name: '${environmentName}-logs'
  location: location
  properties: {
    sku: { name: 'PerGB2018' }
    retentionInDays: 30
  }
}

resource caEnvironment 'Microsoft.App/managedEnvironments@2023-05-01' = {
  name: environmentName
  location: location
  properties: {
    appLogsConfiguration: {
      destination: 'log-analytics'
      logAnalyticsConfiguration: {
        customerId: logAnalytics.properties.customerId
        sharedKey: logAnalytics.listKeys().primarySharedKey
      }
    }
    vnetConfiguration: {
      infrastructureSubnetId: infrastructureSubnetId
      internal: false
    }
  }
}

resource apiApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'rag-api'
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    managedEnvironmentId: caEnvironment.id
    configuration: {
      ingress: {
        external: true
        targetPort: 8000
        transport: 'http'
        traffic: [{ weight: 100, latestRevision: true }]
      }
      secrets: [
        { name: 'azure-openai-api-key', keyVaultUrl: '${kvUri}secrets/azure-openai-api-key', identity: 'system' }
        { name: 'cohere-api-key', keyVaultUrl: '${kvUri}secrets/cohere-api-key', identity: 'system' }
        { name: 'postgres-connection-string', keyVaultUrl: '${kvUri}secrets/postgres-connection-string', identity: 'system' }
        { name: 'redis-connection-string', keyVaultUrl: '${kvUri}secrets/redis-connection-string', identity: 'system' }
        { name: 'langfuse-secret-key', keyVaultUrl: '${kvUri}secrets/langfuse-secret-key', identity: 'system' }
        { name: 'langfuse-public-key', keyVaultUrl: '${kvUri}secrets/langfuse-public-key', identity: 'system' }
      ]
    }
    template: {
      containers: [
        {
          name: 'rag-api'
          image: '${acrLoginServer}/rag-api:${imageTag}'
          resources: { cpu: json('1.0'), memory: '2Gi' }
          env: [
            { name: 'APP_ENV', value: 'production' }
            { name: 'LOG_LEVEL', value: 'INFO' }
            { name: 'AZURE_OPENAI_ENDPOINT', value: azureOpenAIEndpoint }
            { name: 'AZURE_OPENAI_DEPLOYMENT', value: 'gpt-4o' }
            { name: 'AZURE_OPENAI_EMBEDDING_DEPLOYMENT', value: 'text-embedding-3-large' }
            { name: 'AZURE_OPENAI_API_VERSION', value: '2024-10-21' }
            { name: 'EMBEDDING_MODEL', value: 'text-embedding-3-large' }
            { name: 'EMBEDDING_DIMS', value: '3072' }
            { name: 'RERANKER_TOP_K', value: '20' }
            { name: 'FINAL_TOP_K', value: '5' }
            { name: 'CHUNK_SIZE', value: '512' }
            { name: 'PARENT_CHUNK_SIZE', value: '2048' }
            { name: 'CHUNK_OVERLAP', value: '50' }
            { name: 'MAX_CONTEXT_TOKENS', value: '2000' }
            { name: 'CACHE_TTL_SECONDS', value: '3600' }
            { name: 'AZURE_OPENAI_API_KEY', secretRef: 'azure-openai-api-key' }
            { name: 'COHERE_API_KEY', secretRef: 'cohere-api-key' }
            { name: 'DATABASE_URL', secretRef: 'postgres-connection-string' }
            { name: 'REDIS_URL', secretRef: 'redis-connection-string' }
            { name: 'LANGFUSE_SECRET_KEY', secretRef: 'langfuse-secret-key' }
            { name: 'LANGFUSE_PUBLIC_KEY', secretRef: 'langfuse-public-key' }
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
            }
            {
              type: 'Startup'
              httpGet: { path: '/health', port: 8000 }
              initialDelaySeconds: 5
              periodSeconds: 5
              failureThreshold: 12
            }
          ]
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 10
        rules: [
          {
            name: 'http-scaling'
            http: { metadata: { concurrentRequests: '20' } }
          }
        ]
      }
    }
  }
}

resource workerApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'rag-worker'
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    managedEnvironmentId: caEnvironment.id
    configuration: {
      secrets: apiApp.properties.configuration.secrets
    }
    template: {
      containers: [
        {
          name: 'rag-worker'
          image: '${acrLoginServer}/rag-api:${imageTag}'
          command: ['celery']
          args: ['-A', 'src.indexing.worker', 'worker', '--loglevel=info', '-Q', 'ingest,reindex', '--concurrency=4']
          resources: { cpu: json('2.0'), memory: '4Gi' }
          env: apiApp.properties.template.containers[0].env
        }
      ]
      scale: {
        minReplicas: 0
        maxReplicas: 5
        rules: [
          {
            name: 'redis-queue-scaling'
            custom: {
              type: 'redis'
              metadata: {
                listName: 'celery'
                listLength: '5'
                address: redisHostName
              }
            }
          }
        ]
      }
    }
  }
}

resource beatApp 'Microsoft.App/containerApps@2023-05-01' = {
  name: 'rag-beat'
  location: location
  identity: { type: 'SystemAssigned' }
  properties: {
    managedEnvironmentId: caEnvironment.id
    template: {
      containers: [
        {
          name: 'rag-beat'
          image: '${acrLoginServer}/rag-api:${imageTag}'
          command: ['celery']
          args: ['-A', 'src.indexing.worker', 'beat', '--loglevel=info']
          resources: { cpu: json('0.25'), memory: '512Mi' }
          env: apiApp.properties.template.containers[0].env
        }
      ]
      scale: {
        minReplicas: 1
        maxReplicas: 1
      }
    }
  }
}

output apiPrincipalId string = apiApp.identity.principalId
output apiUrl string = apiApp.properties.latestRevisionFqdn
