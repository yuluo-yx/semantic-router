import React from 'react';
import ComponentCreator from '@docusaurus/ComponentCreator';

export default [
  {
    path: '/docs',
    component: ComponentCreator('/docs', 'e87'),
    routes: [
      {
        path: '/docs',
        component: ComponentCreator('/docs', '57f'),
        routes: [
          {
            path: '/docs',
            component: ComponentCreator('/docs', '75e'),
            routes: [
              {
                path: '/docs/api/classification',
                component: ComponentCreator('/docs/api/classification', '048'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/api/router',
                component: ComponentCreator('/docs/api/router', '4e8'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/architecture/envoy-extproc',
                component: ComponentCreator('/docs/architecture/envoy-extproc', 'af5'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/architecture/router-implementation',
                component: ComponentCreator('/docs/architecture/router-implementation', '053'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/architecture/system-architecture',
                component: ComponentCreator('/docs/architecture/system-architecture', 'c91'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/getting-started/configuration',
                component: ComponentCreator('/docs/getting-started/configuration', '468'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/getting-started/installation',
                component: ComponentCreator('/docs/getting-started/installation', '267'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/getting-started/quick-start',
                component: ComponentCreator('/docs/getting-started/quick-start', '09c'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/intro',
                component: ComponentCreator('/docs/intro', '61d'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/overview/mixture-of-models',
                component: ComponentCreator('/docs/overview/mixture-of-models', '51f'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/overview/semantic-router-overview',
                component: ComponentCreator('/docs/overview/semantic-router-overview', '821'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/training/classification-models',
                component: ComponentCreator('/docs/training/classification-models', '5d2'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/training/datasets',
                component: ComponentCreator('/docs/training/datasets', '58a'),
                exact: true,
                sidebar: "tutorialSidebar"
              },
              {
                path: '/docs/training/training-overview',
                component: ComponentCreator('/docs/training/training-overview', '3d9'),
                exact: true,
                sidebar: "tutorialSidebar"
              }
            ]
          }
        ]
      }
    ]
  },
  {
    path: '/',
    component: ComponentCreator('/', '2e1'),
    exact: true
  },
  {
    path: '*',
    component: ComponentCreator('*'),
  },
];
