import { useState, useMemo } from 'react'
import { ChevronRight, ChevronDown, File, Folder, Search } from 'lucide-react'
import type { GraphNode } from '../../api/types'
import { KIND_COLORS } from './constants'

interface TreeNode {
  name: string
  path: string
  kind: 'directory' | 'file' | 'symbol'
  nodeId?: string // graph node ID
  graphKind?: string
  children: TreeNode[]
}

interface Props {
  nodes: GraphNode[]
  onSelectNode: (nodeId: string) => void
  selectedNodeId: string | null
}

const SKIP_KINDS = new Set([
  'memory', 'chunk', 'commit', 'pull_request', 'external',
])

/** Build a hierarchical tree from flat graph nodes. */
function buildTree(nodes: GraphNode[]): TreeNode[] {
  const root: TreeNode = { name: '', path: '', kind: 'directory', children: [] }

  const fileNodes = nodes.filter((n) => n.kind === 'file')
  const symbolNodes = nodes.filter(
    (n) => !SKIP_KINDS.has(n.kind) && n.kind !== 'file' && n.kind !== 'package' && n.kind !== 'module',
  )

  // Build directory structure from file paths
  const filesByPath = new Map<string, GraphNode>()
  for (const node of fileNodes) {
    const path = node.label || node.id.replace(/^file:/, '')
    filesByPath.set(path, node)
  }

  // Insert file nodes into tree
  for (const [path, node] of filesByPath) {
    insertPath(root, path, node.id, node.kind)
  }

  // Attach symbols to their parent files
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

    // Fallback: if no file nodes exist, build tree from symbol module paths.
    // Symbol IDs look like "sym:cli::Commands::Analyze" — use "::" segments as tree levels.
    if (filesByPath.size === 0) {
      const symPath = sym.id.replace(/^sym:/, '')
      const segments = symPath.split('::')
      if (segments.length > 1) {
        // Use all but last segment as directory path, last as the symbol name
        const modulePath = segments.slice(0, -1).join('/')
        const symName = segments[segments.length - 1]
        // Ensure the module directory exists
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
        // Single-segment symbol — add to root
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

  // Sort: directories first, then files, then alphabetical
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
  for (const child of node.children) {
    sortTree(child)
  }
}

function TreeItem({
  node,
  depth,
  onSelect,
  selectedId,
}: {
  node: TreeNode
  depth: number
  onSelect: (nodeId: string) => void
  selectedId: string | null
}) {
  const [expanded, setExpanded] = useState(depth < 2)
  const hasChildren = node.children.length > 0
  const isSelected = node.nodeId === selectedId

  const handleClick = () => {
    if (node.nodeId) {
      onSelect(node.nodeId)
    }
    if (hasChildren) {
      setExpanded(!expanded)
    }
  }

  const icon =
    node.kind === 'directory' ? (
      <Folder size={14} className="shrink-0 text-zinc-500" />
    ) : node.kind === 'file' ? (
      <File size={14} className="shrink-0 text-zinc-500" />
    ) : (
      <span
        className="inline-block h-2.5 w-2.5 shrink-0 rounded-full"
        style={{ backgroundColor: KIND_COLORS[node.graphKind ?? ''] ?? '#71717a' }}
      />
    )

  return (
    <>
      <button
        onClick={handleClick}
        className={`flex w-full items-center gap-1.5 rounded px-1.5 py-0.5 text-left text-xs transition-colors ${
          isSelected
            ? 'bg-violet-500/15 text-violet-300'
            : 'text-zinc-400 hover:bg-zinc-800 hover:text-zinc-200'
        }`}
        style={{ paddingLeft: `${depth * 12 + 4}px` }}
      >
        {hasChildren ? (
          expanded ? (
            <ChevronDown size={12} className="shrink-0 text-zinc-600" />
          ) : (
            <ChevronRight size={12} className="shrink-0 text-zinc-600" />
          )
        ) : (
          <span className="w-3 shrink-0" />
        )}
        {icon}
        <span className="truncate">{node.name}</span>
      </button>
      {expanded &&
        node.children.map((child) => (
          <TreeItem
            key={child.path}
            node={child}
            depth={depth + 1}
            onSelect={onSelect}
            selectedId={selectedId}
          />
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
      const filteredChildren = node.children
        .map(filterNode)
        .filter((c): c is TreeNode => c !== null)
      if (filteredChildren.length > 0) {
        return { ...node, children: filteredChildren }
      }
      return null
    }
    return tree.map(filterNode).filter((c): c is TreeNode => c !== null)
  }, [tree, filter])

  return (
    <div className="flex h-full flex-col">
      <div className="border-b border-zinc-800 p-2">
        <div className="flex items-center gap-1.5 rounded-md border border-zinc-700 bg-zinc-900 px-2 py-1">
          <Search size={12} className="text-zinc-500" />
          <input
            type="text"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter files..."
            className="w-full bg-transparent text-xs text-zinc-300 outline-none placeholder:text-zinc-600"
          />
        </div>
      </div>
      <div className="flex-1 overflow-y-auto p-1">
        {filteredTree.length === 0 ? (
          <p className="p-3 text-center text-xs text-zinc-600">No files found</p>
        ) : (
          filteredTree.map((node) => (
            <TreeItem
              key={node.path}
              node={node}
              depth={0}
              onSelect={onSelectNode}
              selectedId={selectedNodeId}
            />
          ))
        )}
      </div>
    </div>
  )
}
