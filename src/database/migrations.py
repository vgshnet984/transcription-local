from sqlalchemy import create_engine, Column, Integer, String, DateTime, text
from sqlalchemy.orm import sessionmaker, declarative_base
from datetime import datetime
from config import settings

Base = declarative_base()

class MigrationVersion(Base):
    __tablename__ = "migration_version"
    id = Column(Integer, primary_key=True)
    version = Column(String(50), unique=True, nullable=False)
    applied_at = Column(DateTime, default=datetime.utcnow)

def get_engine():
    return create_engine(settings.database_url)

def get_session():
    engine = get_engine()
    Session = sessionmaker(bind=engine)
    return Session()

def ensure_version_table():
    engine = get_engine()
    if not engine.dialect.has_table(engine, "migration_version"):
        Base.metadata.create_all(engine)

def get_current_version():
    session = get_session()
    ensure_version_table()
    row = session.query(MigrationVersion).order_by(MigrationVersion.id.desc()).first()
    session.close()
    return row.version if row else None

def apply_migration(version: str, sql: str):
    engine = get_engine()
    ensure_version_table()
    with engine.connect() as conn:
        conn.execute(text(sql))
        conn.execute(
            MigrationVersion.__table__.insert().values(version=version, applied_at=datetime.utcnow())
        )
        conn.commit()
    print(f"Migration {version} applied.")

# Example usage: add a column to audio_files
if __name__ == "__main__":
    # Example: add 'description' column to audio_files
    version = "20240701_add_description_to_audio_files"
    sql = """
    ALTER TABLE audio_files ADD COLUMN description TEXT;
    """
    current = get_current_version()
    if current != version:
        apply_migration(version, sql)
    else:
        print(f"Migration {version} already applied.") 