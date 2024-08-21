Getting Started
===============

ssh into the Gadi system 

.. code-block:: console
    :linenos:

    ssh <nci user name>@gadi.nci.org.au

or you can use the Gadi terminal option `ARE <https://are.nci.org.au>`_.

change directory to the project directory:

.. code-block:: console
    :linenos:

    cd /scratch/vp91

create a directory in your username:

.. code-block:: console
    :linenos:

    mkdir $USER 
    cd $USER 



To clone the repository, use the following commands:

.. code-block:: console
    :linenos:

    git clone https://github.com/NCI900-Training-Organisation/intro-to-parallel-programming.git
    git checkout python-dev


To access the Gadi system, follow these steps:

1. **SSH into the Gadi system**:

    .. code-block:: console
        :linenos:

        ssh <nci user name>@gadi.nci.org.au


    Alternatively, you can use the Gadi terminal option at `ARE <https://are.nci.org.au>`_.

2. **Change to the project directory**:

    .. code-block:: console
        :linenos:

        cd /scratch/vp91
    

3. **Create and navigate to a directory with your username**:

    .. code-block:: console
        :linenos:

        mkdir $USER
        cd $USER
   

4. **Clone the repository**:

    .. code-block:: console
        :linenos:

        git clone https://github.com/NCI900-Training-Organisation/intro-to-parallel-programming.git
        git checkout python-dev
  


In the repository:

- The `python/src` directory contains all the Python code.
- The `python/job_scripts` directory includes all the PBS job scripts.
- The `python/job_scripts/sample_outputs` directory holds the sample outputs.
