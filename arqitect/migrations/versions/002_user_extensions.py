"""002 User extensions — preferences and per-user nerve permissions.

Revision ID: 002
Create Date: 2026-03-11
"""

from alembic import op
import sqlalchemy as sa

revision = "002"
down_revision = "001"
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "user_preferences",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(100), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("key", sa.String(200), nullable=False),
        sa.Column("value", sa.Text, nullable=False),
        sa.UniqueConstraint("user_id", "key", name="uq_user_pref"),
    )

    op.create_table(
        "user_facts",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.String(100), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("key", sa.String(200), nullable=False),
        sa.Column("value", sa.Text, nullable=False),
        sa.UniqueConstraint("user_id", "key", name="uq_user_fact"),
    )


def downgrade():
    op.drop_table("user_facts")
    op.drop_table("user_preferences")
