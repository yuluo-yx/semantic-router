import React from 'react';
import clsx from 'clsx';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import Layout from '@theme/Layout';
import HomepageFeatures from '@site/src/components/HomepageFeatures';

import styles from './index.module.css';

function HomepageHeader() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <header className={clsx('hero hero--primary', styles.heroBanner)}>
      <div className="container">
        <div className={styles.heroContent}>
          <div className={styles.heroLeft}>
            <div className={styles.heroTitle}>
              <img
                src="/img/vllm.png"
                alt="vLLM Logo"
                className={styles.vllmLogo}
              />
              <h1 className="hero__title">vLLM Semantic Router</h1>
            </div>
            <p className="hero__subtitle">
              Intelligent <strong>Auto Reasoning</strong> Router for Efficient LLM Inference on <strong>Mixture-of-Models</strong>
            </p>
          </div>
          <div className={styles.heroRight}>
            <img
              src="/img/code.png"
              alt="Code Example"
              className={styles.codeImage}
            />
          </div>
        </div>
        <div className={styles.buttons}>
          <Link
            className="button button--secondary button--lg"
            to="/docs/intro">
            Get Started - 5min ⏱️
          </Link>
        </div>
      </div>
    </header>
  );
}

function FlowDiagram() {
  return (
    <section className={styles.flowSection}>
      <div className="container">
        <div className={styles.architectureContainer}>
          <h2 className={styles.architectureTitle}>Intent Aware Semantic Router Architecture</h2>
          <div className={styles.architectureImageWrapper}>
            <img
              src="/img/architecture.png"
              alt="Intent Aware Semantic Router Architecture"
              className={styles.architectureImage}
            />
          </div>
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title={`${siteConfig.title}`}
      description="Intelligent Mixture-of-Models Router that Understands of the Request's Intention">
      <HomepageHeader />
      <main>
        <FlowDiagram />
        <div className={styles.connectionSection}>
          <div className={styles.connectionLines}>
            <div className={`${styles.connectionLine} ${styles.connectionLine1}`}></div>
            <div className={`${styles.connectionLine} ${styles.connectionLine2}`}></div>
            <div className={`${styles.connectionLine} ${styles.connectionLine3}`}></div>
          </div>
        </div>
        <HomepageFeatures />
      </main>
    </Layout>
  );
}
