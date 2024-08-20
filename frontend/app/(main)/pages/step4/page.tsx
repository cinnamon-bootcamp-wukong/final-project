'use client';
import { Button } from 'primereact/button';
import { Tooltip } from 'primereact/tooltip';
import { useState } from 'react';
import { MenuItem } from 'primereact/menuitem';
import { Steps } from 'primereact/steps'
import { useRouter } from 'next/navigation';
import { InputTextarea } from "primereact/inputtextarea";

const Step4 = () => {
    const [activeIndex, setActiveIndex] = useState<number>(3);
    const [value, setValue] = useState('');
    const items: MenuItem[] = [
        {
            label: 'Step 1'
        },
        {
            label: 'Step 2'
        },
        {
            label: 'Step 3'
        },
        {
            label: 'Step 4'
        },
        {
            label: 'Step 5'
        },
    ];

    const router = useRouter();
    const NextClick = async () => {
        const prompt = {
            value: value
        };
        console.log(JSON.stringify(prompt))
        router.push('step5');
    };
    const BackClick = async () => {
        router.push('step3');
    };

    return (
        <div className="grid p-fluid input-demo">
            <div className="col-12">
                <div className="md:col-12">
                        <Steps model={items} activeIndex={activeIndex} onSelect={(e) => setActiveIndex(e.index)} />
                </div>
                <div className="card">
                    <h5>Give your prompt</h5>
                    <InputTextarea
                        value={value} onChange={(e) => setValue(e.target.value)}
                        placeholder="Your Message"
                        rows={6}
                        cols={30}
                    />
                </div>
                <div className="flex justify-content-center gap-2">
                    <Tooltip target=".back" content="Back to previous step" position="top" />
                    <Button 
                        label="Back" 
                        className="back" 
                        style={{ margin: '0.25em 0.25em', width: '150px' }} 
                        onClick={BackClick}
                    />
                    <Tooltip target=".next" content="Next step" position="top" />
                    <Button 
                        label="Next" 
                        className="next" 
                        style={{ margin: '0.25em 0.25em', width: '150px' }} 
                        onClick={NextClick}
                    />
                </div>
            </div>
        </div>
    );
};

export default Step4;