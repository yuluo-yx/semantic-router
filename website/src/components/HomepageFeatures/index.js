import React from 'react';
import clsx from 'clsx';
import styles from './styles.module.css';

const FeatureList = [
  {
    title: '🎯 Intelligent Model Routing',
    description: (
      <>
        Built in ModernBERT Fine-Tuned Model, to achieve Auto-Reasoning and Auto-Selection of Models
      </>
    ),
  },
  {
    title: '🛡️ Enterprise Security',
    description: (
      <>
        Built in PII detection and Prompt guard, avoiding sending jailbreak prompts to the LLM so as to prevent the LLM from misbehaving.
      </>
    ),
  },
  {
    title: '⚡ Similarity Cache',
    description: (
      <>
        Cache the semantic representation of the prompt so as to reduce the number of prompt tokens and improve the overall inference latency.
      </>
    ),
  },
];

function Feature({title, description}) {
  return (
    <div className={clsx('col col--4')}>
      <div className={clsx('card', styles.featureCard)}>
        <div className="text--center padding-horiz--md">
          <h3 className={styles.featureTitle}>{title}</h3>
          <p className={styles.featureDescription}>{description}</p>
        </div>
      </div>
    </div>
  );
}

export default function HomepageFeatures() {
  return (
    <section className={styles.features}>
      <div className="container">
        <div className="row">
          {FeatureList.map((props, idx) => (
            <Feature key={idx} {...props} />
          ))}
        </div>
      </div>
    </section>
  );
}
