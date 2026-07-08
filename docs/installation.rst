Installation
============

Local development install:

.. code-block:: console

   python -m pip install -e ".[dev]"

Install optional surfaces as needed:

.. code-block:: console

   python -m pip install -e ".[documents,scale,export,datasets,structured,local,cloud,mcp]"
   python -m pip install -e ".[docs]"

Run the gateway locally:

.. code-block:: console

   mixle-serve

Run the chat UI:

.. code-block:: console

   cd frontend
   npm install
   npm run dev

Docker compose smoke run:

.. code-block:: console

   cp deploy/.env.example deploy/.env
   docker compose -f deploy/docker-compose.yml --env-file deploy/.env up -d gateway ollama
   curl localhost:8000/v1/models

Build this documentation:

.. code-block:: console

   make -C docs html
