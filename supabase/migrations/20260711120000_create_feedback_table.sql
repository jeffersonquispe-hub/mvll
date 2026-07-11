-- Tabla de feedback enviado desde la ventana "Sobre el proyecto" del frontend.
create table if not exists public.feedback (
    id bigint generated always as identity primary key,
    message text not null,
    created_at timestamptz not null default now()
);

-- RLS activado sin políticas: solo accesible con la service_role key (la que usa
-- el backend), no desde la anon key pública.
alter table public.feedback enable row level security;
