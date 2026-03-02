import { useEffect, useState } from 'react'
import { FileCode, Files, Code } from 'lucide-react'
import { subscribeIndexing } from '../../api/sse'

interface IndexingState {
  files_scanned: number
  files_parsed: number
  total_symbols: number
  current_file: string
}

export function IndexProgress() {
  const [state, setState] = useState<IndexingState | null>(null)
  const [active, setActive] = useState(false)

  useEffect(() => {
    const unsubscribe = subscribeIndexing((data) => {
      setState(data)
      setActive(true)
    })
    return unsubscribe
  }, [])

  if (!active || !state) return null

  const progress =
    state.files_scanned > 0
      ? Math.round((state.files_parsed / state.files_scanned) * 100)
      : 0

  return (
    <div className="rounded-lg border border-zinc-800 bg-zinc-850 p-4">
      <div className="mb-3 flex items-center justify-between">
        <h4 className="text-sm font-medium text-zinc-300">Indexing in progress</h4>
        <span className="text-xs tabular-nums text-zinc-400">{progress}%</span>
      </div>

      {/* Progress bar */}
      <div className="mb-4 h-1.5 overflow-hidden rounded-full bg-zinc-800">
        <div
          className="h-full rounded-full bg-violet-500 transition-all duration-300"
          style={{ width: `${progress}%` }}
        />
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-3 text-center">
        <div>
          <Files size={14} className="mx-auto mb-1 text-zinc-500" />
          <p className="text-sm font-medium tabular-nums text-zinc-200">
            {state.files_scanned}
          </p>
          <p className="text-xs text-zinc-500">Scanned</p>
        </div>
        <div>
          <FileCode size={14} className="mx-auto mb-1 text-zinc-500" />
          <p className="text-sm font-medium tabular-nums text-zinc-200">
            {state.files_parsed}
          </p>
          <p className="text-xs text-zinc-500">Parsed</p>
        </div>
        <div>
          <Code size={14} className="mx-auto mb-1 text-zinc-500" />
          <p className="text-sm font-medium tabular-nums text-zinc-200">
            {state.total_symbols}
          </p>
          <p className="text-xs text-zinc-500">Symbols</p>
        </div>
      </div>

      {/* Current file */}
      {state.current_file && (
        <p className="mt-3 truncate text-xs text-zinc-500">
          {state.current_file}
        </p>
      )}
    </div>
  )
}
