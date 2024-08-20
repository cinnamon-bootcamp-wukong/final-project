import { Metadata } from 'next';
import Layout from '../../layout/layout';

interface AppLayoutProps {
    children: React.ReactNode;
}

export const metadata: Metadata = {
    title: 'AniFace',
    description: 'Anime-Style Avatar Generation',
    robots: { index: false, follow: false },
    viewport: { initialScale: 1, width: 'device-width' },
    openGraph: {
        type: 'website',
        title: 'AniFace',
        description: 'Anime-Style Avatar Generation',
        ttl: 604800
    },
    icons: {
        icon: '/layout/images/anya.png'
    }
};

export default function AppLayout({ children }: AppLayoutProps) {
    return <Layout>{children}</Layout>;
}