# Kubernetes : kind en local, AKS sur Azure

La même API est déployée sur Kubernetes avec un chart Helm, `k8s/pull-events/`. Testé sur **kind** (Kubernetes dans Docker, sur le poste, 0 €). Le déploiement sur **AKS** (Azure Kubernetes Service) est écrit en Terraform et vérifié par un `terraform plan` à blanc, mais le cluster **n'a jamais été créé** : aucun crédit Azure, règle zéro dépense (voir [AKS](#aks)).

## Le chart

| Objet | Rôle | Choix |
|---|---|---|
| `Deployment` | Lance l'image `ghcr.io/clementrbl/pull-events`, un conteneur par pod | Probes `/health` (startup, readiness, liveness), `requests` 250 m CPU / 512 Mi, `limits` 1 CPU / 1 Gi, utilisateur non root (uid 1000), aucune capacité Linux |
| `ConfigMap` | Variables non secrètes (fournisseur, modèles, prix, URL Langfuse) | Injectée par `envFrom` ; son empreinte est dans les annotations du pod, qui redémarre quand elle change |
| `Secret` `pull-events-secrets` | `DATABASE_URL`, `API_KEY`, `AZURE_OPENAI_API_KEY`, clés Langfuse | kind : créé à la main depuis le Key Vault. AKS : rempli par le pilote CSI (ci-dessous) |
| `Service` | Adresse interne stable vers les pods | `ClusterIP`, port 80 → 8000 |
| `Ingress` | Entrée HTTP(S) par nom d'hôte | Classe `nginx` ; sur AKS, annotation cert-manager pour un certificat Let's Encrypt |
| `HorizontalPodAutoscaler` | 1 à 3 pods selon le CPU (cible 70 %) | Demande `metrics-server` |
| `NetworkPolicy` | Entrée : seulement depuis le namespace `ingress-nginx`, port 8000. Sortie : DNS, 443 (Azure OpenAI, Langfuse, Key Vault), 5432 (PostgreSQL) | Appliquée par le CNI : Azure sur AKS ; kind avec son CNI par défaut ne l'applique pas |
| `SecretProviderClass` (AKS) | Décrit les secrets Key Vault à lire et le Secret Kubernetes à remplir | Pilote CSI Secrets Store, identité managée de l'add-on |

Valeurs : `values.yaml` (défauts), `values-kind.yaml` (local), `values-aks.yaml` (Azure).

## kind : 0 €

Prérequis : Docker, `kind` 0.33, `kubectl` 1.35, `helm` 3.20 (binaires dans `~/.local/bin`).

```bash
export DOCKER_CONTEXT=default
kind create cluster --config k8s/kind/cluster.yaml          # ports 80/443 du poste, sur 127.0.0.1 seulement → ingress
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/controller-v1.15.1/deploy/static/provider/kind/deploy.yaml
kubectl apply -f https://github.com/kubernetes-sigs/metrics-server/releases/latest/download/components.yaml
kubectl patch -n kube-system deployment metrics-server --type=json \
  -p '[{"op":"add","path":"/spec/template/spec/containers/0/args/-","value":"--kubelet-insecure-tls"}]'
kubectl create namespace pull-events
```

Le Secret est créé depuis le Key Vault **par fichiers**, jamais en argument de commande (historique du shell, liste des processus) :

```bash
umask 077; T=$(mktemp -d)
for pair in DATABASE_URL:database-url API_KEY:api-key AZURE_OPENAI_API_KEY:azure-openai-api-key \
            LANGFUSE_PUBLIC_KEY:langfuse-public-key LANGFUSE_SECRET_KEY:langfuse-secret-key; do
  az keyvault secret download --vault-name kv-pull-events-cr24 -n "${pair#*:}" --file "$T/v" -o none
  printf '%s=%s\n' "${pair%%:*}" "$(tr -d '\n' < "$T/v")" >> "$T/secrets.env"
done
kubectl -n pull-events create secret generic pull-events-secrets --from-env-file="$T/secrets.env"
rm -rf "$T"

helm upgrade --install pull-events ./k8s/pull-events -n pull-events -f k8s/pull-events/values-kind.yaml --wait
```

Résultat du 25/10/2026 (le pod local parle à Azure OpenAI et à la base Azure, l'IP du poste étant autorisée) :

```
$ kubectl -n pull-events get pods,svc,ingress,hpa,networkpolicy
pod/pull-events-74788d56cf-45gjm   1/1   Running
service/pull-events                ClusterIP   10.96.165.41   80/TCP
ingress.../pull-events             nginx   pull-events.localtest.me   localhost   80
horizontalpodautoscaler/pull-events   Deployment/pull-events   cpu: 0%/70%   1   3   1
networkpolicy/pull-events

$ curl http://pull-events.localtest.me/health          → 200 {"status":"ok"} en 26 ms
$ curl -X POST .../ask (sans clé)                      → 401
$ curl -X POST .../ask -H @en-tête-avec-la-clé         → 200, 5 sources, 6,6 s, 0,0009 $
```

(`localtest.me` et ses sous-domaines résolvent vers `127.0.0.1`.)

Nettoyer : `kind delete cluster --name pull-events`.

## AKS

**Non créé, par décision du 25/10/2026** : l'abonnement n'a aucun crédit et la règle du projet est zéro dépense (≈ 0,067 €/h, voir [terraform.md](terraform.md#aks--prouvé-par-un-plan-pas-par-un-apply)). Le code est écrit et prouvé par un `terraform plan` à blanc (23 ressources à créer, rien d'inattendu) ; kind reste la preuve d'exécution. Ce que créerait `enable_aks = true` dans `infra/terraform/aks.tf` :

- plan de contrôle **Free**, un nœud `Standard_B2als_v2` (2 vCPU, 4 Go), disque OS 32 Go ;
- Azure CNI en mode *overlay* (les pods ont leurs adresses dans `10.244.0.0/16`, le sous-réseau `snet-aks` ne porte que les nœuds) et `network_policy = azure` pour appliquer la NetworkPolicy ;
- add-on **Secrets Store CSI** : son identité managée reçoit *Key Vault Secrets User* sur chacun des cinq secrets de l'application, et le `SecretProviderClass` du chart les recopie dans le Secret Kubernetes lu par `envFrom` ;
- une IP publique Standard avec l'étiquette DNS `pull-events-cr24.francecentral.cloudapp.azure.com`, utilisée par ingress-nginx et validée par Let's Encrypt (HTTP-01) via cert-manager ;
- le cluster est dans le VNet : PostgreSQL est joint par le point de terminaison privé.

Déploiement, après `terraform apply` :

```bash
az aks get-credentials -g rg-pull-events -n aks-pull-events
helm upgrade --install ingress-nginx ingress-nginx --repo https://kubernetes.github.io/ingress-nginx \
  -n ingress-nginx --create-namespace \
  --set controller.service.annotations."service\.beta\.kubernetes\.io/azure-pip-name"=pip-aks-ingress \
  --set controller.service.annotations."service\.beta\.kubernetes\.io/azure-load-balancer-resource-group"=rg-pull-events
helm upgrade --install cert-manager cert-manager --repo https://charts.jetstack.io \
  -n cert-manager --create-namespace --set crds.enabled=true
kubectl apply -f k8s/cert-manager/cluster-issuer.yaml
kubectl create namespace pull-events
helm upgrade --install pull-events ./k8s/pull-events -n pull-events -f k8s/pull-events/values-aks.yaml \
  --set secrets.keyVault.tenantId=$(terraform -chdir=infra/terraform output -raw tenant_id) \
  --set secrets.keyVault.clientId=$(terraform -chdir=infra/terraform output -raw aks_kv_identity_client_id)
```

Coût : ≈ 0,067 €/h (nœud 0,037, Load Balancer 0,022, IP 0,0044, disque), soit 1,6 € par jour. Détruire après la démonstration : `terraform apply -var enable_aks=false`.

Mesure prévue une fois le cluster créé : `API_KEY=… uv run python -m scripts.evaluate_rag --api-url https://pull-events-cr24.francecentral.cloudapp.azure.com --output-label aks` (codes HTTP et latence vues par un client, voir [docs/evaluation/README.md](../evaluation/README.md)).

## Ce qui change entre Container Apps et Kubernetes

| | Container Apps | Kubernetes (AKS) |
|---|---|---|
| Ce qu'on écrit | Une ressource, quelques options | Sept objets YAML, un chart |
| Mise à zéro sans trafic | Oui (0 réplica) | Non : le nœud tourne, facturé à l'heure |
| Secrets | Références Key Vault natives | Pilote CSI + SecretProviderClass |
| TLS | Fourni | ingress-nginx + cert-manager à installer |
| Réseau entre pods | Géré | NetworkPolicy à écrire |
| Quand le choisir | Une API, un démonstrateur, un petit volume | Plusieurs services, exigences réseau ou outillage standard d'une équipe |
