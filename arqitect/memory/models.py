"""SQLAlchemy models — source of truth for the database schema.

These models define all tables used by Sentient. Alembic reads these
to generate migration scripts.
"""

from sqlalchemy import (
    Column, String, Integer, Float, Text, Boolean, DateTime, JSON,
    ForeignKey, UniqueConstraint, Index, func,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    pass


class Fact(Base):
    __tablename__ = "facts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    category = Column(String(100), nullable=False)
    key = Column(String(200), nullable=False)
    value = Column(Text, nullable=False)
    confidence = Column(Float, default=0.5)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    __table_args__ = (UniqueConstraint("category", "key", name="uq_fact_cat_key"),)


class NerveRegistry(Base):
    __tablename__ = "nerve_registry"
    name = Column(String(200), primary_key=True)
    description = Column(Text, default="")
    role = Column(String(50), default="tool")
    system_prompt = Column(Text, default="")
    examples = Column(JSON, default=list)
    tools = Column(JSON, default=list)
    embedding = Column(JSON, default=list)
    invocations = Column(Integer, default=0)
    successes = Column(Integer, default=0)
    avg_latency = Column(Float, default=0.0)
    is_sense = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class ToolStat(Base):
    __tablename__ = "tool_stats"
    id = Column(Integer, primary_key=True, autoincrement=True)
    tool_name = Column(String(200), nullable=False, unique=True)
    call_count = Column(Integer, default=0)
    success_count = Column(Integer, default=0)
    avg_latency = Column(Float, default=0.0)
    last_error = Column(Text, default="")
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class NerveTool(Base):
    __tablename__ = "nerve_tools"
    id = Column(Integer, primary_key=True, autoincrement=True)
    nerve_name = Column(String(200), ForeignKey("nerve_registry.name"), nullable=False)
    tool_name = Column(String(200), nullable=False)
    __table_args__ = (UniqueConstraint("nerve_name", "tool_name", name="uq_nerve_tool"),)


class QualificationResult(Base):
    __tablename__ = "qualification_results"
    id = Column(Integer, primary_key=True, autoincrement=True)
    nerve_name = Column(String(200), nullable=False)
    score = Column(Float, default=0.0)
    passed = Column(Boolean, default=False)
    test_cases = Column(JSON, default=list)
    improvements = Column(JSON, default=list)
    created_at = Column(DateTime, server_default=func.now())


class Episode(Base):
    __tablename__ = "episodes"
    id = Column(Integer, primary_key=True, autoincrement=True)
    task = Column(Text, nullable=False)
    nerve = Column(String(200), default="")
    tool = Column(String(200), default="")
    success = Column(Boolean, default=True)
    result_summary = Column(Text, default="")
    user_id = Column(String(100), default="")
    created_at = Column(DateTime, server_default=func.now())
    __table_args__ = (Index("idx_episodes_task", "task"),)


class User(Base):
    __tablename__ = "users"
    id = Column(String(100), primary_key=True)
    email = Column(String(200), unique=True, nullable=True)
    display_name = Column(String(200), default="")
    role = Column(String(50), default="user")
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())
    links = relationship("UserLink", back_populates="user")
    preferences = relationship("UserPreference", back_populates="user")


class UserLink(Base):
    __tablename__ = "user_links"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), ForeignKey("users.id"), nullable=False)
    connector = Column(String(50), nullable=False)
    connector_user_id = Column(String(200), nullable=False)
    verified = Column(Boolean, default=False)
    verification_code = Column(String(10), default="")
    created_at = Column(DateTime, server_default=func.now())
    user = relationship("User", back_populates="links")
    __table_args__ = (
        UniqueConstraint("connector", "connector_user_id", name="uq_link_connector"),
        Index("idx_user_links_connector", "connector", "connector_user_id"),
    )


class UserPreference(Base):
    __tablename__ = "user_preferences"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), ForeignKey("users.id"), nullable=False)
    key = Column(String(200), nullable=False)
    value = Column(Text, nullable=False)
    user = relationship("User", back_populates="preferences")
    __table_args__ = (UniqueConstraint("user_id", "key", name="uq_user_pref"),)


class UserFact(Base):
    __tablename__ = "user_facts"
    id = Column(Integer, primary_key=True, autoincrement=True)
    user_id = Column(String(100), ForeignKey("users.id"), nullable=False)
    key = Column(String(200), nullable=False)
    value = Column(Text, nullable=False)
    __table_args__ = (UniqueConstraint("user_id", "key", name="uq_user_fact"),)
