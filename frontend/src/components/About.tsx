'use client'

import { motion } from 'framer-motion'
import { Github, ExternalLink, Book, Cpu, Eye, Zap, Users, FileText, Brain } from 'lucide-react'

const features = [
  {
    icon: Brain,
    title: 'Hierarchical Planning',
    description: 'Breaks complex tasks into sub-goals',
  },
  {
    icon: Eye,
    title: 'Visual Understanding',
    description: 'Screenshot-based page comprehension',
  },
  {
    icon: Cpu,
    title: 'Memory-Augmented',
    description: 'LSTM networks remember past interactions',
  },
  {
    icon: Users,
    title: 'Multi-Task Learning',
    description: 'Trained on diverse web tasks',
  },
  {
    icon: Zap,
    title: 'Exploration',
    description: 'Curiosity-driven discovery of new actions',
  },
  {
    icon: FileText,
    title: 'Auto-Documentation',
    description: 'Generates comprehensive logs and reports',
  },
]

const links = [
  { label: 'GitHub Repository', url: 'https://github.com', icon: Github },
  { label: 'Documentation', url: 'https://docs.example.com', icon: Book },
]

export default function About() {
  return (
    <div className="space-y-8">
      {/* Architecture Section */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 p-6"
      >
        <h3 className="text-lg font-semibold text-white mb-4">🏗️ Architecture</h3>
        <p className="text-sm text-gray-400 mb-4">
          This autonomous browser agent combines cutting-edge technologies:
        </p>
        <ul className="space-y-2 text-sm text-gray-300">
          <li className="flex items-center gap-2">
            <span className="text-cyan-400">▹</span>
            <strong>MAYINI Framework:</strong> Custom deep learning library with neural networks
          </li>
          <li className="flex items-center gap-2">
            <span className="text-cyan-400">▹</span>
            <strong>Vision Transformers:</strong> Visual page understanding without HTML dependency
          </li>
          <li className="flex items-center gap-2">
            <span className="text-cyan-400">▹</span>
            <strong>Playwright:</strong> Cross-browser automation with auto-waiting
          </li>
          <li className="flex items-center gap-2">
            <span className="text-cyan-400">▹</span>
            <strong>Reinforcement Learning:</strong> Policy gradient methods for improvement
          </li>
        </ul>
      </motion.div>

      {/* Key Features */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.1 }}
        className="bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 p-6"
      >
        <h3 className="text-lg font-semibold text-white mb-4">🎯 Key Features</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {features.map((feature) => (
            <div
              key={feature.title}
              className="flex items-start gap-3 p-3 bg-dark-400/30 rounded-lg"
            >
              <div className="w-8 h-8 rounded-lg bg-cyan-500/10 flex items-center justify-center flex-shrink-0">
                <feature.icon className="w-4 h-4 text-cyan-400" />
              </div>
              <div>
                <h4 className="text-sm font-medium text-white">{feature.title}</h4>
                <p className="text-xs text-gray-500">{feature.description}</p>
              </div>
            </div>
          ))}
        </div>
      </motion.div>

      {/* Use Cases */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.2 }}
        className="bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 p-6"
      >
        <h3 className="text-lg font-semibold text-white mb-4">📚 Use Cases</h3>
        <div className="flex flex-wrap gap-2">
          {[
            'Form filling',
            'Web scraping',
            'E-commerce',
            'Navigation',
            'Testing & QA',
            'Data extraction',
          ].map((useCase) => (
            <span
              key={useCase}
              className="px-3 py-1.5 text-sm bg-dark-400/50 rounded-full border border-white/5 text-gray-300"
            >
              {useCase}
            </span>
          ))}
        </div>
      </motion.div>

      {/* Links */}
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ delay: 0.3 }}
        className="bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 p-6"
      >
        <h3 className="text-lg font-semibold text-white mb-4">🔗 Links</h3>
        <div className="space-y-2">
          {links.map((link) => (
            <a
              key={link.label}
              href={link.url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-3 p-3 bg-dark-400/30 rounded-lg hover:bg-dark-400/50 transition-colors group"
            >
              <link.icon className="w-5 h-5 text-gray-400 group-hover:text-cyan-400 transition-colors" />
              <span className="text-sm text-gray-300 group-hover:text-white transition-colors">
                {link.label}
              </span>
              <ExternalLink className="w-4 h-4 text-gray-500 ml-auto group-hover:text-cyan-400 transition-colors" />
            </a>
          ))}
        </div>
      </motion.div>

      {/* License */}
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ delay: 0.4 }}
        className="text-center text-sm text-gray-500"
      >
        MIT License — Free to use and modify!
      </motion.div>
    </div>
  )
}