import React from 'react';
import Layout from '@theme/Layout';
import styles from './work-groups.module.css';

const workingGroups = [
  {
    name: 'Docs',
    description: 'User docs, information architecture, infrastructure',
    label: 'area/document',
    icon: '📚',
    skills: ['Technical writing', 'Documentation tools', 'User experience design'],
    needs: ['API documentation', 'Tutorials', 'Deployment guides']
  },
  {
    name: 'Environment',
    description: 'Docker Compose, Kubernetes, Local support, Cloud Foundry Integration',
    label: 'area/environment',
    icon: '🐳',
    skills: ['Docker', 'Kubernetes', 'Cloud platforms', 'DevOps'],
    needs: ['Helm charts', 'Deployment automation', 'Cloud integrations']
  },
  {
    name: 'RouterCore',
    description: 'Modern-BERT, Classify Algorithm, Auto Reasoning Algorithm',
    label: 'area/core',
    icon: '🧠',
    skills: ['Machine learning', 'BERT models', 'Classification algorithms'],
    needs: ['Model optimization', 'Algorithm improvements', 'Reasoning logic']
  },
  {
    name: 'Networking',
    description: 'Envoy ExtProc, Traffic Management, Networks Optimization',
    label: 'area/networking',
    icon: '🌐',
    skills: ['Envoy proxy', 'Network protocols', 'Performance optimization'],
    needs: ['Load balancing', 'Traffic routing', 'Network security']
  },
  {
    name: 'Bench',
    description: 'Reasoning Benchmark Framework, Performance',
    label: 'area/benchmark',
    icon: '📊',
    skills: ['Performance testing', 'Benchmarking tools', 'Data analysis'],
    needs: ['Benchmark frameworks', 'Performance metrics', 'Testing automation']
  },
  {
    name: 'Test and Release',
    description: 'CI/CD, Build, test, release',
    label: 'area/tooling, area/ci',
    icon: '🔧',
    skills: ['CI/CD tools', 'Build automation', 'Release processes'],
    needs: ['Test automation', 'Release pipelines', 'Quality assurance']
  },
  {
    name: 'User Experience',
    description: 'User experience across vLLM Semantic Router, API, Configuration and CLI guidelines and support',
    label: 'area/user-experience',
    icon: '👥',
    skills: ['API design', 'UX/UI', 'Developer experience'],
    needs: ['API improvements', 'CLI enhancements', 'User feedback integration']
  }
];

function WorkGroupCard({ group, featured = false }) {
  const cardClass = featured
    ? `${styles.workGroupCard} ${styles.featuredWorkGroup}`
    : styles.workGroupCard;

  return (
    <div className={cardClass}>
      <div className={styles.cardHeader}>
        <span className={styles.icon}>{group.icon}</span>
        <h3 className={styles.groupName}>{group.name}</h3>
        <span className={styles.label}>{group.label}</span>
      </div>
      <p className={styles.description}>{group.description}</p>

      {featured ? (
        // Featured layout with side-by-side skills and needs
        <div className={styles.skillsAndNeeds}>
          <div className={styles.skillsSection}>
            <h4>Skills Needed:</h4>
            <ul className={styles.skillsList}>
              {group.skills && group.skills.map((skill, index) => (
                <li key={index}>{skill}</li>
              ))}
            </ul>
          </div>

          <div className={styles.needsSection}>
            <h4>Current Needs:</h4>
            <ul className={styles.needsList}>
              {group.needs && group.needs.map((need, index) => (
                <li key={index}>{need}</li>
              ))}
            </ul>
          </div>
        </div>
      ) : (
        // Regular layout with stacked skills and needs
        <>
          <div className={styles.skillsSection}>
            <h4>Skills Needed:</h4>
            <ul className={styles.skillsList}>
              {group.skills && group.skills.map((skill, index) => (
                <li key={index}>{skill}</li>
              ))}
            </ul>
          </div>

          <div className={styles.needsSection}>
            <h4>Current Needs:</h4>
            <ul className={styles.needsList}>
              {group.needs && group.needs.map((need, index) => (
                <li key={index}>{need}</li>
              ))}
            </ul>
          </div>
        </>
      )}
    </div>
  );
}

export default function WorkGroups() {
  return (
    <Layout
      title="Work Groups"
      description="vLLM Semantic Router Community Working Groups">
      <div className={styles.container}>
        <header className={styles.header}>
          <h1>vLLM Semantic Router Work Groups 👋</h1>
        </header>
        
        <main className={styles.main}>
          <section className={styles.intro}>
            <h2>🌍 WG Initialization</h2>
            <p>
              We are looking for interests around vLLM Semantic Router project and separate it into different WGs.
            </p>
            <p>
              Please comment on{' '}
              <a 
                href="https://github.com/vllm-project/semantic-router/issues/15" 
                target="_blank" 
                rel="noopener noreferrer"
                className={styles.link}
              >
                GitHub Issue #15
              </a>{' '}
              if you are interested in one or more.
            </p>
          </section>

          <section className={styles.workingGroupsSection}>
            <h2>⛰️ vLLM Semantic Router Community WG</h2>
            <p>This section is about setting WG around this project, to gather focus on specify areas.</p>

            <div className={styles.workGroupsGrid}>
              {/* Featured RouterCore group */}
              {workingGroups
                .filter(group => group.name === 'RouterCore')
                .map((group, index) => (
                  <WorkGroupCard key={`featured-${index}`} group={group} featured={true} />
                ))}

              {/* Other working groups */}
              {workingGroups
                .filter(group => group.name !== 'RouterCore')
                .map((group, index) => (
                  <WorkGroupCard key={index} group={group} />
                ))}
            </div>
          </section>

          <section className={styles.promotion}>
            <h2>🔝 Community Promotion</h2>
            <p>
              We are grateful for any contributions, and if you show consistent contributions to the above specify area, 
              you will be promoting as its maintainer after votes from maintainer team, and you will be invited to 
              semantic-router-maintainer group, and granted WRITE access to this repo.
            </p>
          </section>

          <section className={styles.getInvolved}>
            <h2>How to Get Involved</h2>
            <ol className={styles.stepsList}>
              <li><strong>Choose Your Interest Area:</strong> Review the working groups above and identify which areas align with your skills and interests</li>
              <li><strong>Join the Discussion:</strong> Comment on <a href="https://github.com/vllm-project/semantic-router/issues/15" target="_blank" rel="noopener noreferrer">GitHub Issue #15</a> to express your interest</li>
              <li><strong>Start Contributing:</strong> Look for issues labeled with the corresponding area tags (e.g., <code>area/document</code>, <code>area/core</code>)</li>
              <li><strong>Collaborate:</strong> Connect with other community members working in the same areas</li>
            </ol>
          </section>

          <section className={styles.contact}>
            <h2>Contact</h2>
            <p>For questions about working groups or to get involved:</p>
            <ul>
              <li>Open an issue on <a href="https://github.com/vllm-project/semantic-router/issues" target="_blank" rel="noopener noreferrer">GitHub</a></li>
              <li>Join the discussion on <a href="https://github.com/vllm-project/semantic-router/issues/15" target="_blank" rel="noopener noreferrer">Issue #15</a></li>
              <li>Check out our <a href="/docs/intro">documentation</a> to get started</li>
            </ul>
          </section>
        </main>
      </div>
    </Layout>
  );
}
