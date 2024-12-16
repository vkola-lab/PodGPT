# coding=utf-8
#
# GNU Affero General Public License v3.0 License
#
# PodGPT: An Audio-augmented Large Language Model for Research and Education
# Copyright (C) 2024 Kolachalama Laboratory at Boston University

import time

from sqlalchemy import create_engine, text, Column, String
from sqlalchemy.engine import URL
from sqlalchemy.orm import sessionmaker
from sqlalchemy.exc import OperationalError
from sqlalchemy.dialects.postgresql import BIT
from sqlalchemy.ext.declarative import declarative_base
from pgvector.sqlalchemy import Vector, SPARSEVEC

from lib.config import DATABASE_CONFIG

# Configure the database URL using DATABASE_CONFIG from config.py
db_kwargs = {
    'drivername': DATABASE_CONFIG['drivername'],  # Database driver (e.g., postgres)
    'host': DATABASE_CONFIG['host'],  # Database host (e.g., localhost)
    'database': DATABASE_CONFIG['database'],  # Database name
    'username': DATABASE_CONFIG['user'],  # Database username
    'port': DATABASE_CONFIG['port'],  # Database port
    'password': DATABASE_CONFIG['password']  # Database password
}

# Create the database URL
db_url = URL.create(**db_kwargs)

# Create a database engine with specific connection settings
engine = create_engine(
    db_url,
    echo=False,  # Disable SQL query logging
    pool_pre_ping=True,  # Enable connection health check before reuse
    pool_recycle=600,  # Recycle connections after 10 minutes
    connect_args={
        'connect_timeout': 1,  # Timeout for initial connection
        'keepalives': 1,  # Enable TCP keepalives
        'keepalives_idle': 10,  # Time before sending keepalive probes
        'keepalives_interval': 10,  # Interval between keepalive probes
        'keepalives_count': 5,  # Maximum failed keepalive probes before disconnection
    }
)

# Define a base class for SQLAlchemy models
Base = declarative_base()

# Create a session factory bound to the engine
Access_orm = sessionmaker(bind=engine)


# Function to obtain a database session with retry logic
def get_session():
    retries = 5  # Number of connection attempts
    for _ in range(retries):
        try:
            session = Access_orm()
            # Verify the database connection
            session.execute(text('SELECT 1'))
            return session
        except OperationalError:
            # Wait before retrying in case of connection failure
            time.sleep(1)
    # Raise an exception if all attempts fail
    raise Exception("Failed to connect to the database after multiple attempts")


# Define the PMCArticles table model
class PMCArticles(Base):
    __tablename__ = 'pmcarticles'
    __table_args__ = {'extend_existing': True}

    # Define table columns with specific data types
    embedding_gte_qwen2_7b_instruct = Column(Vector(3584))  # Vector embedding
    embedding_name = Column(String, primary_key=True)  # Unique identifier
    pmc_id = Column(String)  # Article ID
    text = Column(String)  # Article text
    binary_embedding = Column(BIT(4096))  # Binary vector for embedding
    sparse_vector = Column(SPARSEVEC(30522))  # Sparse vector representation


# Define the PMCFullArticles table model
class PMCFullArticles(Base):
    __tablename__ = 'pmcfullarticles'
    __table_args__ = {'extend_existing': True}

    # Define table columns
    pmc_id = Column(String, primary_key=True)  # Primary key: article ID
    file_name = Column(String)  # File name of the article
    full_article_text = Column(String)  # Full text of the article
    sparse_vector = Column(SPARSEVEC(30522))  # Sparse vector representation


# Define the PMCMetadata table model
class PMCMetadata(Base):
    __tablename__ = "pmcmetadata"
    __table_args__ = {'extend_existing': True}

    # Define table columns with primary keys and additional metadata
    article_file = Column(String)  # File path to the article
    article_citation = Column(String)  # Citation text
    accessionid = Column(String, primary_key=True)  # Unique accession ID
    lastupdated = Column(String, primary_key=True)  # Timestamp of last update
    pmid = Column(String)  # PubMed ID
    license = Column(String)  # License information
    retracted = Column(String)  # Retraction status
    citation_ama = Column(String)  # Citation in AMA format
    citation_apa = Column(String)  # Citation in APA format
    citation_mla = Column(String)  # Citation in MLA format
    citation_nlm = Column(String)  # Citation in NLM format


# Create all tables based on defined models
Base.metadata.create_all(engine)
