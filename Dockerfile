FROM python:3.12-slim

# uv installe les dépendances depuis uv.lock (versions figées)
COPY --from=ghcr.io/astral-sh/uv:0.10.7 /uv /bin/uv

# Utilisateur non privilégié (audit 3.3). uid 1000 : le même que celui imposé
# par le chart Helm (runAsUser) et que l'utilisateur du poste, propriétaire du
# dossier data/ monté par Docker Compose.
RUN useradd --uid 1000 --user-group --no-create-home --shell /usr/sbin/nologin app

WORKDIR /app
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

# Dépendances d'abord : cette couche reste en cache tant que le lockfile ne change pas
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

# Copier le code source ; data/ accueille les fichiers du pipeline (/rebuild)
COPY --chown=app:app api/ api/
COPY --chown=app:app scripts/ scripts/
RUN mkdir data && chown app:app data

USER app

# Port de l'API
EXPOSE 8000

# Lancer l'API. Pas de --forwarded-allow-ips "*" : l'IP du client est lue
# par l'API elle-même (dernière adresse de X-Forwarded-For, voir api/main.py).
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
