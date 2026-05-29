'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import { Brain, Loader2, ChevronRight, Sparkles } from 'lucide-react'
import { decomposeTask } from '@/lib/api'
import { cn } from '@/lib/utils'

export default function TaskPlanner() {
  const [task, setTask] = useState('')
  const [decomposition, setDecomposition] = useState<string | null>(null)
  const [isLoading, setIsLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const handleDecompose = async () => {
    if (!task.trim()) {
      setError('Please enter a task description')
      return
    }

    setIsLoading(true)
    setError(null)
    setDecomposition(null)

    try {
      const response = await decomposeTask(task)
      setDecomposition(response.decomposition || response.error || 'No decomposition available')
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to decompose task')
    } finally {
      setIsLoading(false)
    }
  }

  // Parse markdown-like decomposition
  const parseDecomposition = (text: string) => {
    const lines = text.split('\n').filter(line => line.trim())
    const items: { type: 'header' | 'item' | 'text'; content: string }[] = []
    
    lines.forEach(line => {
      if (line.startsWith('#')) {
        items.push({ type: 'header', content: line.replace(/^#+\s*/, '') })
      } else if (line.match(/^\d+\./)) {
        items.push({ type: 'item', content: line.replace(/^\d+\.\s*/, '') })
      } else {
        items.push({ type: 'text', content: line })
      }
    })
    
    return items
  }

  const parsedItems = decomposition ? parseDecomposition(decomposition) : []

  return (
    <div className="space-y-6">
      {/* Input Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 p-6"
      >
        <div className="flex items-center gap-2 mb-4">
          <Brain className="w-5 h-5 text-cyan-400" />
          <h3 className="text-lg font-semibold text-white">Task Decomposer</h3>
        </div>
        
        <p className="text-sm text-gray-400 mb-4">
          Enter a complex task to see how it will be broken down into smaller, manageable sub-tasks.
        </p>

        <textarea
          value={task}
          onChange={(e) => {
            setTask(e.target.value)
            setError(null)
          }}
          placeholder="Example: Buy a laptop on Amazon and compare prices"
          rows={3}
          className={cn(
            'w-full px-4 py-3 bg-dark-400/50 rounded-lg border text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-cyan-500/50 transition-all resize-none',
            error ? 'border-rose-500' : 'border-white/10'
          )}
        />

        {error && (
          <p className="mt-2 text-sm text-rose-400">{error}</p>
        )}

        <motion.button
          whileHover={{ scale: 1.02 }}
          whileTap={{ scale: 0.98 }}
          onClick={handleDecompose}
          disabled={isLoading || !task.trim()}
          className={cn(
            'mt-4 px-6 py-3 rounded-lg font-medium text-white transition-all flex items-center gap-2',
            isLoading || !task.trim()
              ? 'bg-gray-600 cursor-not-allowed'
              : 'bg-gradient-to-r from-purple-600 to-cyan-600 hover:from-purple-500 hover:to-cyan-500 shadow-lg'
          )}
        >
          {isLoading ? (
            <>
              <Loader2 className="w-4 h-4 animate-spin" />
              Decomposing...
            </>
          ) : (
            <>
              <Sparkles className="w-4 h-4" />
              Decompose Task
            </>
          )}
        </motion.button>
      </motion.div>

      {/* Results Section */}
      {parsedItems.length > 0 && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          className="bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 p-6"
        >
          <h4 className="text-sm font-medium text-gray-300 mb-4">📋 Decomposition Results</h4>
          
          <div className="space-y-3">
            {parsedItems.map((item, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className={cn(
                  'flex items-start gap-3 p-4 rounded-lg',
                  item.type === 'header' 
                    ? 'bg-gradient-to-r from-cyan-500/10 to-purple-500/10 border border-cyan-500/20'
                    : 'bg-dark-400/30'
                )}
              >
                {item.type === 'item' && (
                  <div className="flex-shrink-0 w-6 h-6 rounded-full bg-cyan-500/20 flex items-center justify-center">
                    <span className="text-xs font-bold text-cyan-400">{index}</span>
                  </div>
                )}
                <span className={cn(
                  'text-sm',
                  item.type === 'header' ? 'text-cyan-400 font-semibold' : 'text-gray-300'
                )}>
                  {item.content}
                </span>
              </motion.div>
            ))}
          </div>

          {/* Visual flow */}
          <div className="mt-6 pt-6 border-t border-white/5">
            <h5 className="text-xs font-medium text-gray-500 mb-4">Workflow</h5>
            <div className="flex items-center gap-2 overflow-x-auto pb-2">
              {parsedItems
                .filter(item => item.type === 'item')
                .map((item, index) => (
                  <div key={index} className="flex items-center">
                    <motion.div
                      initial={{ opacity: 0, scale: 0.8 }}
                      animate={{ opacity: 1, scale: 1 }}
                      transition={{ delay: index * 0.1 }}
                      className="px-4 py-2 bg-dark-400 rounded-lg text-sm text-white whitespace-nowrap"
                    >
                      {item.content.length > 20 
                        ? item.content.substring(0, 20) + '...' 
                        : item.content}
                    </motion.div>
                    {index < parsedItems.filter(i => i.type === 'item').length - 1 && (
                      <ChevronRight className="w-4 h-4 text-gray-500 mx-1" />
                    )}
                  </div>
                ))}
            </div>
          </div>
        </motion.div>
      )}

      {/* Help text */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.3 }}
        className="text-center text-sm text-gray-500"
      >
        💡 Tip: Break down complex tasks into specific, actionable steps for better results.
      </motion.div>
    </div>
  )
}