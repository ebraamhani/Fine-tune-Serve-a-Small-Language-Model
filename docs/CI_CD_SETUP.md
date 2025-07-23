# CI/CD Setup Documentation

This document describes the comprehensive CI/CD pipeline setup for the Fine-tune-Serve-a-Small-Language-Model project.

## Overview

The project uses GitHub Actions for continuous integration and deployment, with multiple workflows covering different aspects of the development lifecycle.

## Workflow Files

### 1. `ci.yml` - Main CI/CD Pipeline

**Triggers:**
- Push to `main` or `develop` branches
- Pull requests to `main` or `develop` branches
- Manual trigger (`workflow_dispatch`)

**Jobs:**
- **Test**: Multi-Python version testing, linting, security checks
- **Security**: Vulnerability scanning with Trivy
- **Build**: Package building and artifact creation
- **Deploy Staging**: Automatic deployment to staging (develop branch)
- **Deploy Production**: Automatic deployment to production (main branch)

### 2. `ml-training.yml` - ML Training Pipeline

**Triggers:**
- Changes to training-related files
- Manual trigger with model/dataset parameters

**Jobs:**
- **Data Validation**: Dataset availability and quality checks
- **Model Training**: Multi-model/multi-dataset training with Weights & Biases
- **Model Evaluation**: Performance evaluation and reporting
- **Model Deployment**: Best model deployment to Hugging Face Hub

### 3. `release.yml` - Release Management

**Triggers:**
- GitHub release creation
- Manual trigger with version parameter

**Jobs:**
- **Create Release**: Package building, PyPI publishing, Hugging Face Hub upload

### 4. `docker.yml` - Container Deployment

**Triggers:**
- Changes to deployment files or Dockerfile
- Manual trigger

**Jobs:**
- **Build and Push**: Docker image building and registry push
- **Deploy Staging**: Container deployment to staging
- **Deploy Production**: Container deployment to production

### 5. `scheduled.yml` - Maintenance Tasks

**Triggers:**
- Daily at 2 AM UTC
- Manual trigger

**Jobs:**
- **Dependency Update**: Security checks and dependency updates
- **Model Health Check**: Hugging Face connectivity and model availability
- **Cleanup**: Artifact cleanup

## Required Secrets

Set up the following secrets in your GitHub repository settings:

### Core Secrets
- `HF_TOKEN`: Your Hugging Face API token
- `WANDB_API_KEY`: Weights & Biases API key for experiment tracking

### Deployment Secrets
- `PYPI_API_TOKEN`: PyPI API token for package publishing
- `DOCKER_REGISTRY_TOKEN`: Docker registry authentication (if using private registry)

### Environment-Specific Secrets
- `STAGING_DEPLOY_KEY`: SSH key for staging deployment
- `PRODUCTION_DEPLOY_KEY`: SSH key for production deployment

## Environment Setup

### GitHub Environments

Create the following environments in your repository:

1. **staging**
   - Protection rules: Require reviewers
   - Deployment branches: `develop`

2. **production**
   - Protection rules: Require reviewers, wait timer
   - Deployment branches: `main`

## Usage Examples

### Manual Workflow Triggers

```bash
# Trigger ML training with specific parameters
gh workflow run ml-training.yml -f model_name=gpt2 -f dataset_name=squad

# Trigger release with version
gh workflow run release.yml -f version=1.2.0

# Trigger manual deployment
gh workflow run docker.yml
```

### Branch Strategy

- **`main`**: Production-ready code, triggers production deployment
- **`develop`**: Integration branch, triggers staging deployment
- **Feature branches**: Create PRs to `develop` for testing

## Customization

### Adding New Models

1. Update the matrix in `ml-training.yml`:
```yaml
strategy:
  matrix:
    model: [gpt2, distilbert-base-uncased, your-new-model]
    dataset: [squad, glue, your-new-dataset]
```

### Adding New Deployment Targets

1. Add new environment in GitHub settings
2. Create new job in relevant workflow file
3. Add environment-specific secrets

### Custom Actions

Create custom actions in `.github/actions/` for reusable components:

```yaml
- name: Custom Action
  uses: ./.github/actions/custom-action
  with:
    parameter: value
```

## Monitoring and Debugging

### Workflow Logs

- View workflow runs in the "Actions" tab
- Download logs for debugging
- Use `actions/upload-artifact` for large files

### Notifications

Configure notifications for:
- Workflow failures
- Security vulnerabilities
- Deployment status

### Performance Optimization

- Use caching for dependencies
- Parallel job execution
- Conditional job execution based on file changes

## Security Considerations

1. **Secrets Management**: Never hardcode secrets in workflow files
2. **Dependency Scanning**: Regular security checks with Safety and Bandit
3. **Container Security**: Trivy vulnerability scanning
4. **Access Control**: Environment protection rules
5. **Audit Logging**: GitHub provides audit logs for all actions

## Troubleshooting

### Common Issues

1. **Permission Denied**: Check repository permissions and secrets
2. **Timeout Errors**: Increase timeout limits or optimize workflow
3. **Dependency Conflicts**: Use virtual environments and pin versions
4. **Resource Limits**: Use self-hosted runners for resource-intensive tasks

### Debug Commands

```bash
# Check workflow syntax
yamllint .github/workflows/*.yml

# Test locally with act (requires Docker)
act -j test

# Validate secrets
gh secret list
```

## Best Practices

1. **Version Pinning**: Always pin action versions for stability
2. **Caching**: Use caching for expensive operations
3. **Parallelization**: Run independent jobs in parallel
4. **Conditional Execution**: Use `if` conditions to avoid unnecessary runs
5. **Artifact Management**: Clean up old artifacts regularly
6. **Documentation**: Keep workflow documentation updated

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [GitHub Actions Marketplace](https://github.com/marketplace?type=actions)
- [GitHub Actions Examples](https://github.com/actions/starter-workflows)
- [Security Best Practices](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions) 