"""Initial database schema for Logistics Email AI.

Revision ID: 0001
Revises: 
Create Date: 2024-01-01 00:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '0001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create initial database schema."""
    
    # Create tenants table
    op.create_table('tenants',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('name', sa.String(length=255), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_tenants_name', 'tenants', ['name'])
    
    # Create threads table
    op.create_table('threads',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('subject_norm', sa.String(length=500), nullable=True),
        sa.Column('first_email_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('last_message_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ),
        sa.ForeignKeyConstraint(['first_email_id'], ['emails.id'], )
    )
    op.create_index('idx_threads_tenant_last_message', 'threads', ['tenant_id', 'last_message_at'], postgresql_ops={'last_message_at': 'DESC'})
    op.create_index('idx_threads_tenant_subject', 'threads', ['tenant_id', 'subject_norm'])
    
    # Create emails table
    op.create_table('emails',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('thread_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('message_id', sa.String(length=500), nullable=False),
        sa.Column('subject', sa.String(length=500), nullable=True),
        sa.Column('from_addr', sa.String(length=255), nullable=True),
        sa.Column('to_addrs', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('cc_addrs', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('bcc_addrs', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('sent_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('received_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('in_reply_to', sa.String(length=500), nullable=True),
        sa.Column('references', postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.Column('snippet', sa.Text(), nullable=True),
        sa.Column('has_attachments', sa.Boolean(), nullable=False, default=False),
        sa.Column('raw_object_key', sa.String(length=500), nullable=True),
        sa.Column('norm_object_key', sa.String(length=500), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.Column('updated_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ),
        sa.ForeignKeyConstraint(['thread_id'], ['threads.id'], ),
        sa.UniqueConstraint('tenant_id', 'message_id', name='uq_emails_tenant_message')
    )
    op.create_index('idx_emails_tenant_thread_sent', 'emails', ['tenant_id', 'thread_id', 'sent_at'], postgresql_ops={'sent_at': 'DESC'})
    op.create_index('idx_emails_tenant_created', 'emails', ['tenant_id', 'created_at'], postgresql_ops={'created_at': 'DESC'})
    op.create_index('idx_emails_tenant_attachments', 'emails', ['tenant_id', 'has_attachments'])
    op.create_index('idx_emails_tenant_from', 'emails', ['tenant_id', 'from_addr'])
    op.create_index('idx_emails_tenant_subject', 'emails', ['tenant_id', 'subject'])
    
    # Create attachments table
    op.create_table('attachments',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('filename', sa.String(length=500), nullable=True),
        sa.Column('mimetype', sa.String(length=100), nullable=True),
        sa.Column('size_bytes', sa.BigInteger(), nullable=True),
        sa.Column('object_key', sa.String(length=500), nullable=True),
        sa.Column('ocr_text_object_key', sa.String(length=500), nullable=True),
        sa.Column('ocr_state', sa.String(length=50), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], )
    )
    op.create_index('idx_attachments_tenant_email', 'attachments', ['tenant_id', 'email_id'])
    op.create_index('idx_attachments_tenant_mimetype', 'attachments', ['tenant_id', 'mimetype'])
    op.create_index('idx_attachments_tenant_size', 'attachments', ['tenant_id', 'size_bytes'])
    
    # Create chunks table
    op.create_table('chunks',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('attachment_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('chunk_uid', sa.String(length=255), nullable=False),
        sa.Column('content', sa.Text(), nullable=False),
        sa.Column('token_count', sa.Integer(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ),
        sa.ForeignKeyConstraint(['attachment_id'], ['attachments.id'], )
    )
    op.create_index('idx_chunks_tenant_email', 'chunks', ['tenant_id', 'email_id'])
    op.create_index('idx_chunks_tenant_uid', 'chunks', ['tenant_id', 'chunk_uid'])
    op.create_index('idx_chunks_tenant_attachment', 'chunks', ['tenant_id', 'attachment_id'])
    op.create_index('idx_chunks_tenant_tokens', 'chunks', ['tenant_id', 'token_count'])
    
    # Create labels table
    op.create_table('labels',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('name', sa.String(length=100), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ),
        sa.UniqueConstraint('tenant_id', 'name', name='uq_labels_tenant_name')
    )
    op.create_index('idx_labels_tenant_name', 'labels', ['tenant_id', 'name'])
    
    # Create email_labels table
    op.create_table('email_labels',
        sa.Column('tenant_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('label_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], ),
        sa.ForeignKeyConstraint(['label_id'], ['labels.id'], ),
        sa.UniqueConstraint('tenant_id', 'email_id', 'label_id', name='uq_email_labels_tenant_email_label')
    )
    op.create_index('idx_email_labels_tenant_email', 'email_labels', ['tenant_id', 'email_id'])
    op.create_index('idx_email_labels_tenant_label', 'email_labels', ['tenant_id', 'label_id'])
    
    # Create eval_runs table
    op.create_table('eval_runs',
        sa.Column('id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('tenant_id', postgresql.UUID(as_uuid=False), nullable=False),
        sa.Column('email_id', postgresql.UUID(as_uuid=False), nullable=True),
        sa.Column('draft_id', sa.String(length=255), nullable=True),
        sa.Column('rubric', sa.String(length=100), nullable=True),
        sa.Column('score_grounding', sa.Float(), nullable=True),
        sa.Column('score_completeness', sa.Float(), nullable=True),
        sa.Column('score_tone', sa.Float(), nullable=True),
        sa.Column('score_policy', sa.Float(), nullable=True),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('now()'), nullable=False),
        sa.PrimaryKeyConstraint('id'),
        sa.ForeignKeyConstraint(['tenant_id'], ['tenants.id'], ),
        sa.ForeignKeyConstraint(['email_id'], ['emails.id'], )
    )
    op.create_index('idx_eval_runs_tenant_email', 'eval_runs', ['tenant_id', 'email_id'])
    op.create_index('idx_eval_runs_tenant_created', 'eval_runs', ['tenant_id', 'created_at'], postgresql_ops={'created_at': 'DESC'})
    op.create_index('idx_eval_runs_tenant_scores', 'eval_runs', ['tenant_id', 'score_grounding', 'score_completeness', 'score_tone', 'score_policy'])


def downgrade() -> None:
    """Drop all tables."""
    op.drop_table('eval_runs')
    op.drop_table('email_labels')
    op.drop_table('labels')
    op.drop_table('chunks')
    op.drop_table('attachments')
    op.drop_table('emails')
    op.drop_table('threads')
    op.drop_table('tenants')
