'use client';
import React, { useState, useEffect } from 'react';
import Step1 from '../step1/step1'
import Step2 from '../step2/step2'
import Step3 from '../step3/step3'


export default function mainPage() {
    const [activeIndex, setActiveIndex] = useState<number>(0);
    const [pass2Step, setPass2Step] = useState<boolean>(false);

    useEffect(() => {
        console.log(activeIndex);
    }, [activeIndex]);

    return (
        <div>
            <div style={{ display: activeIndex === 0 ? 'block' : 'none' }}>
                <Step1 setStep={setActiveIndex}/>
            </div>
            <div style={{ display: activeIndex === 1 ? 'block' : 'none' }}>
                <Step2 setStep={setActiveIndex} setBool={setPass2Step}/>
            </div>
            <div style={{ display: activeIndex === 2 ? 'block' : 'none' }}>
                <Step3 setStep={setActiveIndex} pass2Step={pass2Step}/>
            </div>
        </div>
    );
}