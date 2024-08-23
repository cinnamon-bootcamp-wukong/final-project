import Link from 'next/link';
import React, { forwardRef, useImperativeHandle, useRef } from 'react';
import { AppTopbarRef } from '@/types';

const AppTopbar = forwardRef<AppTopbarRef>((props, ref) => {
    const menubuttonRef = useRef(null);
    const topbarmenuRef = useRef(null);
    const topbarmenubuttonRef = useRef(null);

    useImperativeHandle(ref, () => ({
        menubutton: menubuttonRef.current,
        topbarmenu: topbarmenuRef.current,
        topbarmenubutton: topbarmenubuttonRef.current
    }));

    return (
        <div className="layout-topbar">
            <Link href="/" className="layout-topbar-logo">
                <img src={`/layout/images/logo.png`} width="60px" height={'35px'} alt="logo" />
            </Link>
        </div>
    );
});

AppTopbar.displayName = 'AppTopbar';

export default AppTopbar;
