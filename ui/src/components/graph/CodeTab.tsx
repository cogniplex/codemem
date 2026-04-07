import { useEffect, useRef, useState } from 'react'
import { Loader2 } from 'lucide-react'
import { useFileContent } from '../../api/hooks'
import { useNamespaceStore } from '../../stores/namespace'
import type { GraphNode } from '../../api/types'

interface Props {
  node: GraphNode
}

const EXT_TO_LANG: Record<string, string> = {
  rs: 'rust',
  ts: 'typescript',
  tsx: 'tsx',
  js: 'javascript',
  jsx: 'jsx',
  py: 'python',
  go: 'go',
  java: 'java',
  rb: 'ruby',
  cs: 'csharp',
  kt: 'kotlin',
  swift: 'swift',
  php: 'php',
  scala: 'scala',
  tf: 'hcl',
  toml: 'toml',
  json: 'json',
  yaml: 'yaml',
  yml: 'yaml',
  md: 'markdown',
  css: 'css',
  html: 'html',
  sql: 'sql',
  sh: 'bash',
  zsh: 'bash',
  c: 'c',
  cpp: 'cpp',
  h: 'c',
  hpp: 'cpp',
}

export function CodeTab({ node }: Props) {
  const namespace = useNamespaceStore((s) => s.active)
  const filePath = (node.payload?.file_path as string) ?? extractFilePath(node)
  const lineStart = (node.payload?.line_start as number) ?? undefined
  const lineEnd = (node.payload?.line_end as number) ?? undefined

  const { data, isLoading, error } = useFileContent(filePath, undefined, undefined, namespace ?? undefined)
  const codeRef = useRef<HTMLDivElement>(null)
  const [highlightedHtml, setHighlightedHtml] = useState<string | null>(null)

  useEffect(() => {
    if (!data?.content) return

    const ext = filePath?.split('.').pop() ?? ''
    const lang = EXT_TO_LANG[ext] ?? 'text'

    // Dynamically import shiki to avoid blocking initial load
    import('shiki').then(async ({ createHighlighter }) => {
      try {
        const highlighter = await createHighlighter({
          themes: ['github-dark'],
          langs: [lang],
        })
        const html = highlighter.codeToHtml(data.content, {
          lang,
          theme: 'github-dark',
        })
        setHighlightedHtml(html)
      } catch {
        // Fallback: plain text
        setHighlightedHtml(null)
      }
    })
  }, [data?.content, filePath])

  // Scroll to highlighted line after render
  useEffect(() => {
    if (!codeRef.current || !lineStart) return
    requestAnimationFrame(() => {
      const el = codeRef.current?.querySelector(`[data-line="${lineStart}"]`)
      el?.scrollIntoView({ block: 'center' })
    })
  }, [highlightedHtml, lineStart])

  if (!filePath) {
    return (
      <div className="flex h-32 items-center justify-center text-xs text-zinc-600">
        No source file associated with this node
      </div>
    )
  }

  if (isLoading) {
    return (
      <div className="flex h-32 items-center justify-center">
        <Loader2 size={14} className="animate-spin text-zinc-500" />
        <span className="ml-2 text-xs text-zinc-500">Loading source...</span>
      </div>
    )
  }

  if (error || !data) {
    return (
      <div className="flex h-32 items-center justify-center text-xs text-zinc-500">
        Could not load file content
      </div>
    )
  }

  // Breadcrumb
  const pathParts = filePath.split('/')

  // Split content into lines for line numbers + highlighting
  const lines = data.content.split('\n')
  const startLine = data.line_start

  return (
    <div className="flex flex-col">
      {/* Breadcrumb */}
      <div className="flex items-center gap-1 border-b border-zinc-800 px-3 py-2 text-xs text-zinc-500">
        {pathParts.map((part, i) => (
          <span key={i} className="flex items-center gap-1">
            {i > 0 && <span className="text-zinc-700">/</span>}
            <span className={i === pathParts.length - 1 ? 'text-zinc-300' : ''}>
              {part}
            </span>
          </span>
        ))}
      </div>

      {/* Code */}
      <div ref={codeRef} className="max-h-80 overflow-auto text-xs">
        {highlightedHtml ? (
          <HighlightedCode
            html={highlightedHtml}
            startLine={startLine}
            highlightStart={lineStart}
            highlightEnd={lineEnd}
          />
        ) : (
          <table className="w-full border-collapse">
            <tbody>
              {lines.map((line, i) => {
                const lineNum = startLine + i
                const isHighlighted =
                  lineStart !== undefined &&
                  lineEnd !== undefined &&
                  lineNum >= lineStart &&
                  lineNum <= lineEnd
                return (
                  <tr
                    key={i}
                    data-line={lineNum}
                    className={isHighlighted ? 'bg-violet-500/10' : ''}
                  >
                    <td className="select-none px-3 py-0 text-right font-mono text-zinc-600">
                      {lineNum}
                    </td>
                    <td className="whitespace-pre px-3 py-0 font-mono text-zinc-300">
                      {line}
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}

function HighlightedCode({
  html,
  startLine,
  highlightStart,
  highlightEnd,
}: {
  html: string
  startLine: number
  highlightStart?: number
  highlightEnd?: number
}) {
  // Extract lines from Shiki's output — they are wrapped in <span class="line">
  const lineRegex = /<span class="line">(.*?)<\/span>/g
  const lineMatches: string[] = []
  let match
  while ((match = lineRegex.exec(html)) !== null) {
    lineMatches.push(match[1])
  }

  // Fallback if regex didn't match Shiki's format
  if (lineMatches.length === 0) {
    return (
      <div
        className="p-3 font-mono [&_pre]:!bg-transparent [&_code]:!bg-transparent"
        dangerouslySetInnerHTML={{ __html: html }}
      />
    )
  }

  return (
    <table className="w-full border-collapse">
      <tbody>
        {lineMatches.map((lineHtml, i) => {
          const lineNum = startLine + i
          const isHighlighted =
            highlightStart !== undefined &&
            highlightEnd !== undefined &&
            lineNum >= highlightStart &&
            lineNum <= highlightEnd
          return (
            <tr
              key={i}
              data-line={lineNum}
              className={isHighlighted ? 'bg-violet-500/10' : ''}
              style={isHighlighted ? { borderLeft: '2px solid rgb(139 92 246)' } : undefined}
            >
              <td className="select-none px-3 py-0 text-right font-mono text-zinc-600">
                {lineNum}
              </td>
              <td
                className="whitespace-pre px-3 py-0 font-mono"
                dangerouslySetInnerHTML={{ __html: lineHtml }}
              />
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

/** Extract file path from node ID conventions like "file:path" or "sym:path::name" */
function extractFilePath(node: GraphNode): string | null {
  if (node.id.startsWith('file:')) {
    return node.id.slice(5)
  }
  // For symbol nodes, the label is the file path in the graph
  if (node.kind === 'file') {
    return node.label
  }
  return null
}
