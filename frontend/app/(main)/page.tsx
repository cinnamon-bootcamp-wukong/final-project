'use client';

import { Button } from 'primereact/button';
import React from 'react';
import { useRouter } from 'next/navigation';
  



const Home = () => {
    const router = useRouter();
    return (
        <div className="grid grid-nogutter surface-0 text-800">
            <div className="col-12 md:col-7 p-6 text-center md:text-left flex align-items-center ">
                <section>
                    <span className="block text-6xl font-bold mb-1">GEN AI</span>
                    <div className="text-6xl text-primary font-bold mb-3">Anime-Style Avatar Generation</div>
                    <p className="mt-0 mb-4 text-700 line-height-3">A generative AI system that transforms human portrait images into anime-style avatars, capable of exhibiting different emotions.</p>

                    <Button label="TRY IT" type="button" className="mr-3 p-button-raised" 
                        onClick={() => {
                            router.push('pages/step');
                        }}
                    />
                </section>
            </div>
            <div className="col-12 md:col-5 overflow-hidden">
                <img 
                    src="/demo/images/blocks/hero/alime.png" 
                    alt="alime" 
                    className="md:ml-auto block" 
                    style={{ 
                        clipPath: 'polygon(8% 0, 100% 0%, 100% 100%, 0 100%)', 
                        width: '70%',    /* Adjust width here */
                        height: 'auto'    /* Adjust height here */
                    }} 
                />
            </div>
        </div>
    );
};

export default Home;
