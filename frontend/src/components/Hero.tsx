'use client'

import { motion } from 'framer-motion'
import { Bot, Zap, Globe, Sparkles, Brain, Monitor } from 'lucide-react'

const features = [
  {
    icon: Brain,
    title: 'MAYINI Framework',
    description: 'Custom deep learning for intelligent decision-making',
  },
  {
    icon: Globe,
    title: 'Vision Transformers',
    description: 'Visual page understanding without HTML dependency',
  },
  {
    icon: Monitor,
    title: 'Playwright',
    description: 'Cross-browser automation with auto-waiting',
  },
  {
    icon: Zap,
    title: 'Reinforcement Learning',
    description: 'Continuous improvement through policy gradients',
  },
]

export default function Hero() {
  return (
    <section className="relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 bg-gradient-to-br from-primary-600/20 via-dark-400 to-dark-500" />
      <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,_var(--tw-gradient-stops))] from-cyan-500/10 via-transparent to-transparent" />
      
      {/* Animated orbs */}
      <motion.div
        animate={{
          scale: [1, 1.2, 1],
          opacity: [0.3, 0.5, 0.3],
        }}
        transition={{
          duration: 8,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
        className="absolute top-20 left-1/4 w-64 h-64 bg-cyan-500/20 rounded-full blur-3xl"
      />
      <motion.div
        animate={{
          scale: [1.2, 1, 1.2],
          opacity: [0.3, 0.5, 0.3],
        }}
        transition={{
          duration: 10,
          repeat: Infinity,
          ease: 'easeInOut',
        }}
        className="absolute bottom-20 right-1/4 w-72 h-72 bg-purple-500/20 rounded-full blur-3xl"
      />

      <div className="relative max-w-6xl mx-auto px-4 py-16 sm:px-6 sm:py-24 lg:px-8">
        {/* Logo and title */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6 }}
          className="flex items-center justify-center mb-8"
        >
          <div className="relative">
            <motion.div
              animate={{ rotate: 360 }}
              transition={{ duration: 20, repeat: Infinity, ease: 'linear' }}
              className="absolute -inset-4 bg-gradient-to-r from-cyan-500 to-purple-500 rounded-full opacity-30 blur-lg"
            />
            <div className="relative bg-gradient-to-br from-dark-300 to-dark-400 p-4 rounded-2xl border border-cyan-500/30 shadow-2xl shadow-cyan-500/10">
              <Bot className="w-12 h-12 text-cyan-400" />
            </div>
          </div>
        </motion.div>

        <motion.h1
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.1 }}
          className="text-4xl sm:text-5xl lg:text-6xl font-bold text-center mb-4"
        >
          <span className="bg-gradient-to-r from-white via-cyan-100 to-white bg-clip-text text-transparent">
            Autonomous Browser Agent
          </span>
        </motion.h1>

        <motion.p
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.2 }}
          className="text-lg sm:text-xl text-gray-400 text-center max-w-2xl mx-auto mb-8"
        >
          Intelligent web automation powered by{' '}
          <span className="text-cyan-400 font-semibold">MAYINI Framework</span>, 
          vision transformers, and reinforcement learning.
        </motion.p>

        {/* Feature pills */}
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.6, delay: 0.3 }}
          className="flex flex-wrap justify-center gap-3 mb-12"
        >
          {['🧠 Deep Learning', '👁️ Visual AI', '🔄 RL Training', '🌐 Web Automation'].map((feature, i) => (
            <motion.span
              key={feature}
              initial={{ opacity: 0, scale: 0.8 }}
              animate={{ opacity: 1, scale: 1 }}
              transition={{ duration: 0.4, delay: 0.3 + i * 0.1 }}
              className="px-4 py-2 bg-dark-300/80 backdrop-blur-sm rounded-full text-sm text-gray-300 border border-white/10"
            >
              {feature}
            </motion.span>
          ))}
        </motion.div>

        {/* Features grid */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.5 }}
          className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4"
        >
          {features.map((feature, i) => (
            <motion.div
              key={feature.title}
              initial={{ opacity: 0, y: 20 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.4, delay: 0.5 + i * 0.1 }}
              className="group p-6 bg-dark-300/50 backdrop-blur-sm rounded-xl border border-white/5 hover:border-cyan-500/30 transition-all duration-300"
            >
              <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-cyan-500/20 to-purple-500/20 flex items-center justify-center mb-4 group-hover:scale-110 transition-transform">
                <feature.icon className="w-5 h-5 text-cyan-400" />
              </div>
              <h3 className="font-semibold text-white mb-2">{feature.title}</h3>
              <p className="text-sm text-gray-400">{feature.description}</p>
            </motion.div>
          ))}
        </motion.div>

        {/* Stats */}
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.6, delay: 0.7 }}
          className="mt-12 flex flex-wrap justify-center gap-8"
        >
          {[
            { value: '50+', label: 'Actions' },
            { value: '99%', label: 'Accuracy' },
            { value: '24/7', label: 'Monitoring' },
          ].map((stat, i) => (
            <div key={stat.label} className="text-center">
              <div className="text-3xl font-bold bg-gradient-to-r from-cyan-400 to-purple-400 bg-clip-text text-transparent">
                {stat.value}
              </div>
              <div className="text-sm text-gray-500">{stat.label}</div>
            </div>
          ))}
        </motion.div>
      </div>
    </section>
  )
}