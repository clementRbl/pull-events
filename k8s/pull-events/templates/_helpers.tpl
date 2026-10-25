{{- define "pull-events.name" -}}
{{ .Chart.Name }}
{{- end }}

{{- define "pull-events.labels" -}}
app.kubernetes.io/name: {{ include "pull-events.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{- define "pull-events.selectorLabels" -}}
app.kubernetes.io/name: {{ include "pull-events.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}
