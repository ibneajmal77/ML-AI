targetScope = 'resourceGroup'

param environmentName string = 'dev'
param location string = resourceGroup().location

param registryName string = 'ragkbregistry${environmentName}'
param keyVaultName string = 'rag-kb-kv-${environmentName}'
param postgresServerName string = 'rag-kb-postgres-${environmentName}'
param redisCacheName string = 'rag-kb-redis-${environmentName}'
param containerAppsEnvName string = 'rag-kb-env-${environmentName}'
param vnetName string = 'rag-kb-vnet-${environmentName}'
param logAnalyticsWorkspaceName string = 'rag-kb-logs-${environmentName}'

@secure()
param postgresAdminPassword string

param azureOpenAIEndpoint string
param acrSku string = 'Basic'
param postgresSku string = 'Standard_D2ds_v5'

module network 'modules/network.bicep' = {
  name: 'network'
  params: {
    location: location
    vnetName: vnetName
  }
}

module registry 'modules/registry.bicep' = {
  name: 'registry'
  params: {
    location: location
    registryName: registryName
    sku: acrSku
  }
}

module postgres 'modules/postgres.bicep' = {
  name: 'postgres'
  params: {
    location: location
    serverName: postgresServerName
    adminPassword: postgresAdminPassword
    subnetId: network.outputs.dataSubnetId
    privateDnsZoneId: network.outputs.postgresDnsZoneId
  }
  dependsOn: [network]
}

module redis 'modules/redis.bicep' = {
  name: 'redis'
  params: {
    location: location
    cacheName: redisCacheName
    subnetId: network.outputs.dataSubnetId
  }
  dependsOn: [network]
}

module keyVault 'modules/keyvault.bicep' = {
  name: 'keyVault'
  params: {
    location: location
    keyVaultName: keyVaultName
  }
}

module containerApps 'modules/container-apps.bicep' = {
  name: 'containerApps'
  params: {
    location: location
    environmentName: containerAppsEnvName
    infrastructureSubnetId: network.outputs.appsSubnetId
    acrLoginServer: registry.outputs.loginServer
    postgresServerFqdn: postgres.outputs.postgresServerFqdn
    redisHostName: redis.outputs.redisHostName
    kvUri: keyVault.outputs.keyVaultUri
    azureOpenAIEndpoint: azureOpenAIEndpoint
  }
  dependsOn: [network, registry, postgres, redis, keyVault]
}

// Reference the vault by name so we can scope the role assignment to it.
// This sits outside both modules to break the circular dependency.
resource kvExisting 'Microsoft.KeyVault/vaults@2023-07-01' existing = {
  name: keyVaultName
}

resource kvSecretUserRole 'Microsoft.Authorization/roleAssignments@2022-04-01' = {
  scope: kvExisting
  name: guid(kvExisting.id, containerApps.outputs.apiPrincipalId, 'Key Vault Secrets User')
  properties: {
    roleDefinitionId: subscriptionResourceId(
      'Microsoft.Authorization/roleDefinitions',
      '4633458b-17de-408a-b874-0445c86b69e6'
    )
    principalId: containerApps.outputs.apiPrincipalId
    principalType: 'ServicePrincipal'
  }
}

output acrLoginServer string = registry.outputs.loginServer
output postgresServerFqdn string = postgres.outputs.postgresServerFqdn
output redisHostName string = redis.outputs.redisHostName
output keyVaultUri string = keyVault.outputs.keyVaultUri
