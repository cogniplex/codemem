import { useState, useMemo } from 'react'
import { ChevronRight, ChevronDown, File, Folder, Search } from 'lucide-react'
import type { GraphNode } from '../../api/types'
import { KIND_COLORS } from './constants'

interface TreeNode {
  name: string
  path: string
  kind: 'directory' | 'file' | 'symbol'
  nodeId?: string
  graphKind?: string
  children: TreeNode[]
}

interface Props {
  nodes: GraphNode[]
  onSelectNode: (nodeId: string) => void
  selectedNodeId: string | null
}

const SKIP_KINDS = new Set(['memory', 'chunk', 'commit', 'pull_request', 'external'])

// File extension → color for file icons
const EXT_COLORS: Record<string, string> = {
  rs: 'text-orange-400',
  ts: 'text-blue-400',
  tsx: 'text-blue-400',
  js: 'text-yellow-400',
  jsx: 'text-yellow-400',
  py: 'text-green-400',
  go: 'text-cyan-400',
  java: 'text-red-400',
  rb: 'text-red-400',
  toml: 'text-zinc-400',
  json: 'text-amber-400',
  md: 'text-zinc-500',
  yml: 'text-pink-400',
  yaml: 'text-pink-400',
  css: 'text-violet-400',
  html: 'text-orange-400',
  sql: 'text-cyan-400',
}

function getExtColor(name: string): string {
  const ext = name.split('.').pop()?.toLowerCase() ?? ''
  return EXT_COLORS[ext] ?? 'text-zinc-500'
}

function buildTree(nodes: GraphNode[]): TreeNode[] {
  const root: TreeNode = { name: '', path: '', kind: 'directory', children: [] }
  const fileNodes = nodes.filter((n) => n.kind === 'file')
  const symbolNodes = nodes.filter(
    (n) => !SKIP_KINDS.has(n.kind) && n.kind !== 'file' && n.kind !== 'package' && n.kind !== 'module',
  )

  const filesByPath = new Map<string, GraphNode>()
  for (const node of fileNodes) {
    const path = node.label || node.id.replace(/^file:/, '')
    filesByPath.set(path, node)
  }

  for (const [path, node] of filesByPath) {
    insertPath(root, path, node.id, node.kind)
  }

  for (const sym of symbolNodes) {
    const filePath = sym.payload?.file_path as string | undefined
    if (filePath && filesByPath.has(filePath)) {
      const fileTreeNode = findTreeNode(root, `file:${filePath}`) ?? findTreeNode(root, filePath)
      if (fileTreeNode) {
        fileTreeNode.children.push({
          name: sym.label,
          path: `${filePath}#${sym.label}`,
          kind: 'symbol',
          nodeId: sym.id,
          graphKind: sym.kind,
          children: [],
        })
        continue
      }
    }

    if (filesByPath.size === 0) {
      const symPath = sym.id.replace(/^sym:/, '')
      const segments = symPath.split('::')
      if (segments.length > 1) {
        const modulePath = segments.slice(0, -1).join('/')
        const symName = segments[segments.length - 1]
        insertPath(root, modulePath, undefined, undefined)
        const parentDir = findTreeNodeByPath(root, modulePath)
        if (parentDir) {
          parentDir.children.push({
            name: symName,
            path: `${modulePath}#${symName}`,
            kind: 'symbol',
            nodeId: sym.id,
            graphKind: sym.kind,
            children: [],
          })
        }
      } else {
        root.children.push({
          name: sym.label,
          path: `#${sym.label}`,
          kind: 'symbol',
          nodeId: sym.id,
          graphKind: sym.kind,
          children: [],
        })
      }
    }
  }

  sortTree(root)
  return root.children
}

function insertPath(root: TreeNode, path: string, nodeId?: string, graphKind?: string) {
  const parts = path.split('/')
  let current = root
  for (let i = 0; i < parts.length; i++) {
    const part = parts[i]
    const isLast = i === parts.length - 1
    let child = current.children.find((c) => c.name === part)
    if (!child) {
      child = {
        name: part,
        path: parts.slice(0, i + 1).join('/'),
        kind: isLast && nodeId ? 'file' : 'directory',
        nodeId: isLast ? nodeId : undefined,
        graphKind: isLast ? graphKind : undefined,
        children: [],
      }
      current.children.push(child)
    }
    current = child
  }
}

function findTreeNodeByPath(node: TreeNode, path: string): TreeNode | null {
  if (node.path === path) return node
  for (const child of node.children) {
    const found = findTreeNodeByPath(child, path)
    if (found) return found
  }
  return null
}

function findTreeNode(node: TreeNode, nodeId: string): TreeNode | null {
  if (node.nodeId === nodeId) return node
  for (const child of node.children) {
    const found = findTreeNode(child, nodeId)
    if (found) return found
  }
  return null
}

function sortTree(node: TreeNode) {
  node.children.sort((a, b) => {
    if (a.kind === 'directory' && b.kind !== 'directory') return -1
    if (a.kind !== 'directory' && b.kind === 'directory') return 1
    return a.name.localeCompare(b.name)
  })
  for (const child of node.children) sortTree(child)
}

function TreeItem({
  node, depth, onSelect, selectedId,
}: {
  node: TreeNode; depth: number; onSelect: (nodeId: string) => void; selectedId: string | null
}) {
  const [expanded, setExpanded] = useState(depth < 1)
  const hasChildren = node.children.length > 0
  const isSelected = node.nodeId === selectedId

  const handleClick = () => {
    if (node.nodeId) onSelect(node.nodeId)
    if (hasChildren) setExpanded(!expanded)
  }

  return (
    <>
      <button
        onClick={handleClick}
        className={`flex w-full items-center gap-1.5 rounded-md py-[3px] text-left font-mono text-[12px] leading-snug transition-colors ${
          isSelected
            ? 'bg-violet-500/10 text-violet-300'
            : 'text-zinc-400 hover:bg-zinc-800/50 hover:text-zinc-200'
        }`}
        style={{ paddingLeft: `${depth * 14 + 8}px`, paddingRight: '8px' }}
      >
        {hasChildren ? (
          expanded ? (
            <ChevronDown size={10} className="shrink-0 text-zinc-600" />
          ) : (
            <ChevronRight size={10} className="shrink-0 text-zinc-600" />
          )
        ) : (
          <span className="w-[10px] shrink-0" />
        )}
        {node.kind === 'directory' ? (
          <Folder size={12} className={`shrink-0 ${expanded ? 'text-zinc-400' : 'text-zinc-600'}`} />
        ) : node.kind === 'file' ? (
          <File size={12} className={`shrink-0 ${getExtColor(node.name)}`} />
        ) : (
          <span
            className="inline-block h-[7px] w-[7px] shrink-0 rounded-sm"
            style={{ backgroundColor: KIND_COLORS[node.graphKind ?? ''] ?? '#52525b' }}
          />
        )}
        <span className="truncate">{node.name}</span>
        {hasChildren && !expanded && (
          <span className="ml-auto shrink-0 text-[9px] tabular-nums text-zinc-700">
            {node.children.length}
          </span>
        )}
      </button>
      {expanded && node.children.map((child) => (
        <TreeItem key={child.path} node={child} depth={depth + 1} onSelect={onSelect} selectedId={selectedId} />
      ))}
    </>
  )
}

export function FileTree({ nodes, onSelectNode, selectedNodeId }: Props) {
  const [filter, setFilter] = useState('')
  const tree = useMemo(() => buildTree(nodes), [nodes])

  const filteredTree = useMemo(() => {
    if (!filter) return tree
    const lower = filter.toLowerCase()
    function filterNode(node: TreeNode): TreeNode | null {
      if (node.name.toLowerCase().includes(lower)) return node
      const fc = node.children.map(filterNode).filter((c): c is TreeNode => c !== null)
      if (fc.length > 0) return { ...node, children: fc }
      return null
    }
    return tree.map(filterNode).filter((c): c is TreeNode => c !== null)
  }, [tree, filter])

  return (
    <div className="flex h-full flex-col">
      <div className="px-2 py-2">
        <div className="flex items-center gap-1.5 rounded-md border border-zinc-800/50 bg-zinc-900/50 px-2 py-[5px]">
          <Search size={11} className="text-zinc-600" />
          <input
            type="text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter..."
            className="w-full bg-transparent text-[12px] text-zinc-300 outline-none placeholder:text-zinc-700"
          />
        </div>
      </div>
      <div className="flex-1 overflow-y-auto px-1 pb-2">
        {filteredTree.length === 0 ? (
          <p className="px-3 py-6 text-center text-[11px] text-zinc-700">No files</p>
        ) : (
          filteredTree.map((node) => (
            <TreeItem key={node.path} node={node} depth={0} onSelect={onSelectNode} selectedId={selectedNodeId} />
          ))
        )}
      </div>
    </div>
  )
}
