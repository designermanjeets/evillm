"""Add embedding_jobs table

Revision ID: 0002
Revises: 0001
Create Date: 2025-08-16 18:30:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0002'
down_revision = '0001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Create embedding_jobs table
    op.create_table('embedding_jobs',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('chunk_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('status', sa.String(length=50), nullable=False),
        sa.Column('priority', sa.Integer(), nullable=False),
        sa.Column('retry_count', sa.Integer(), nullable=False),
        sa.Column('max_retries', sa.Integer(), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('started_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('completed_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('error_message', sa.Text(), nullable=True),
        sa.Column('embedding_vector', postgresql.VECTOR(1536), nullable=True),
        sa.Column('metadata', postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default='{}'),
        sa.ForeignKeyConstraint(['chunk_id'], ['chunks.id'], ),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ),
        sa.PrimaryKeyConstraint('id'),
        sa.CheckConstraint("status IN ('pending', 'processing', 'completed', 'failed')", name='valid_embedding_job_status')
    )
    
    # Create indexes for performance
    op.create_index('idx_embedding_jobs_tenant_chunk', 'embedding_jobs', ['tenant_id', 'chunk_id'])
    op.create_index('idx_embedding_jobs_tenant_email', 'embedding_jobs', ['tenant_id', 'email_id'])
    op.create_index('idx_embedding_jobs_tenant_status', 'embedding_jobs', ['tenant_id', 'status'])
    op.create_index('idx_embedding_jobs_tenant_priority', 'embedding_jobs', ['tenant_id', 'priority'])
    op.create_index('idx_embedding_jobs_tenant_retry_count', 'embedding_jobs', ['tenant_id', 'retry_count'])
    op.create_index('idx_embedding_jobs_tenant_started_at', 'embedding_jobs', ['tenant_id', 'started_at'])
    op.create_index('idx_embedding_jobs_tenant_completed_at', 'embedding_jobs', ['tenant_id', 'completed_at'])


def downgrade() -> None:
    # Drop indexes
    op.drop_index('idx_embedding_jobs_tenant_completed_at', table_name='embedding_jobs')
    op.drop_index('idx_embedding_jobs_tenant_started_at', table_name='embedding_jobs')
    op.drop_index('idx_embedding_jobs_tenant_retry_count', table_name='embedding_jobs')
    op.drop_index('idx_embedding_jobs_tenant_priority', table_name='embedding_jobs')
    op.drop_index('idx_embedding_jobs_tenant_status', table_name='embedding_jobs')
    op.drop_index('idx_embedding_jobs_tenant_email', table_name='embedding_jobs')
    op.drop_index('idx_embedding_jobs_tenant_chunk', table_name='embedding_jobs')
    
    # Drop table
    op.drop_table('embedding_jobs')
