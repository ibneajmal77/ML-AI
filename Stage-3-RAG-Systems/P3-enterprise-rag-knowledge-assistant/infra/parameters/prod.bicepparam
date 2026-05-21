using '../main.bicep'

param environmentName = 'prod'
param acrSku = 'Standard'
param postgresAdminPassword = '' // Set via az deployment with --parameters postgresAdminPassword=... or Key Vault reference
param azureOpenAIEndpoint = 'https://your-resource.openai.azure.com/'
