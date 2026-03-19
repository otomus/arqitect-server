"""001 Initial schema.

Revision ID: 001
Create Date: 2026-03-11
"""

from alembic import op
import sqlalchemy as sa

revision = "001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    # Facts
    op.create_table(
        "facts",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("category", sa.String(100), nullable=False),
        sa.Column("key", sa.String(200), nullable=False),
        sa.Column("value", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float, default=0.5),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint("category", "key", name="uq_fact_cat_key"),
    )

    # Nerve registry
    op.create_table(
        "nerve_registry",
        sa.Column("name", sa.String(200), primary_key=True),
        sa.Column("description", sa.Text, default=""),
        sa.Column("role", sa.String(50), default="tool"),
        sa.Column("system_prompt", sa.Text, default=""),
        sa.Column("examples", sa.JSON, default=[]),
        sa.Column("tools", sa.JSON, default=[]),
        sa.Column("embedding", sa.JSON, default=[]),
        sa.Column("invocations", sa.Integer, default=0),
        sa.Column("successes", sa.Integer, default=0),
        sa.Column("avg_latency", sa.Float, default=0.0),
        sa.Column("is_sense", sa.Boolean, default=False),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    # Tool stats
    op.create_table(
        "tool_stats",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("tool_name", sa.String(200), nullable=False, unique=True),
        sa.Column("call_count", sa.Integer, default=0),
        sa.Column("success_count", sa.Integer, default=0),
        sa.Column("avg_latency", sa.Float, default=0.0),
        sa.Column("last_error", sa.Text, default=""),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    # Nerve tools junction
    op.create_table(
        "nerve_tools",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("nerve_name", sa.String(200), sa.ForeignKey("nerve_registry.name"), nullable=False),
        sa.Column("tool_name", sa.String(200), nullable=False),
        sa.UniqueConstraint("nerve_name", "tool_name", name="uq_nerve_tool"),
    )

    # Qualification results
    op.create_table(
        "qualification_results",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("nerve_name", sa.String(200), nullable=False),
        sa.Column("score", sa.Float, default=0.0),
        sa.Column("passed", sa.Boolean, default=False),
        sa.Column("test_cases", sa.JSON, default=[]),
        sa.Column("improvements", sa.JSON, default=[]),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )

    # Episodes (warm memory)
    op.create_table(
        "episodes",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("task", sa.Text, nullable=False),
        sa.Column("nerve", sa.String(200), default=""),
        sa.Column("tool", sa.String(200), default=""),
        sa.Column("success", sa.Boolean, default=True),
        sa.Column("result_summary", sa.Text, default=""),
        sa.Column("user_id", sa.String(100), default=""),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
    )
    op.create_index("idx_episodes_task", "episodes", ["task"])

    # Users
    op.create_table(
        "users",
        sa.Column("id", sa.String(100), primary_key=True),
        sa.Column("email", sa.String(200), unique=True, nullable=True),
        sa.Column("display_name", sa.String(200), default=""),
        sa.Column("role", sa.String(50), default="user"),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime, server_default=sa.func.now()),
    )

    # User links (connector identity mapping)
    op.create_table(
        "user_links",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(100), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("connector", sa.String(50), nullable=False),
        sa.Column("connector_user_id", sa.String(200), nullable=False),
        sa.Column("verified", sa.Boolean, default=False),
        sa.Column("verification_code", sa.String(10), default=""),
        sa.Column("created_at", sa.DateTime, server_default=sa.func.now()),
        sa.UniqueConstraint("connector", "connector_user_id", name="uq_link_connector"),
    )
    op.create_index("idx_user_links_connector", "user_links", ["connector", "connector_user_id"])


def downgrade():
    op.drop_table("user_links")
    op.drop_table("users")
    op.drop_table("episodes")
    op.drop_table("qualification_results")
    op.drop_table("nerve_tools")
    op.drop_table("tool_stats")
    op.drop_table("nerve_registry")
    op.drop_table("facts")
