// SSE event source clients for real-time updates

export function subscribeIndexing(
  onProgress: (data: { files_scanned: number; files_parsed: number; total_symbols: number; current_file: string }) => void,
): () => void {
  const source = new EventSource('/api/events/indexing')
  source.addEventListener('indexing', (e) => {
    try {
      const data = JSON.parse(e.data)
      onProgress(data)
    } catch { /* ignore parse errors */ }
  })
  return () => source.close()
}

export function subscribeWatch(
  onEvent: (data: { path: string; event_type: string; timestamp: string }) => void,
): () => void {
  const source = new EventSource('/api/events/watch')
  source.addEventListener('watch', (e) => {
    try {
      const data = JSON.parse(e.data)
      onEvent(data)
    } catch { /* ignore parse errors */ }
  })
  return () => source.close()
}

export interface AgentRunCallbacks {
  onRecipeStart?: (data: { recipe: string; name: string; total_steps: number }) => void
  onStepStart?: (data: { step: number; tool: string; description: string }) => void
  onStepResult?: (data: { step: number; tool: string; success: boolean; result: string }) => void
  onComplete?: (data: { recipe: string }) => void
  onError?: (data: { error: string }) => void
}

export function runRecipe(
  body: { recipe: string; repo_id?: string; namespace?: string },
  callbacks: AgentRunCallbacks,
): () => void {
  const ctrl = new AbortController()

  fetch('/api/agents/run', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', Accept: 'text/event-stream' },
    body: JSON.stringify(body),
    signal: ctrl.signal,
  })
    .then(async (res) => {
      if (!res.ok || !res.body) {
        callbacks.onError?.({ error: `HTTP ${res.status}` })
        return
      }
      const reader = res.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ''

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })

        const lines = buffer.split('\n')
        buffer = lines.pop() ?? ''

        let currentEvent = ''
        for (const line of lines) {
          if (line.startsWith('event: ')) {
            currentEvent = line.slice(7).trim()
          } else if (line.startsWith('data: ')) {
            const data = line.slice(6)
            try {
              const parsed = JSON.parse(data)
              switch (currentEvent) {
                case 'recipe_start':
                  callbacks.onRecipeStart?.(parsed)
                  break
                case 'step_start':
                  callbacks.onStepStart?.(parsed)
                  break
                case 'step_result':
                  callbacks.onStepResult?.(parsed)
                  break
                case 'recipe_complete':
                  callbacks.onComplete?.(parsed)
                  break
                case 'error':
                  callbacks.onError?.(parsed)
                  break
              }
            } catch { /* ignore parse errors */ }
            currentEvent = ''
          }
        }
      }
    })
    .catch((err) => {
      if (err.name !== 'AbortError') {
        callbacks.onError?.({ error: String(err) })
      }
    })

  return () => ctrl.abort()
}
