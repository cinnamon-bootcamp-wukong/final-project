'use client';
import { Button } from 'primereact/button';
import { Tooltip } from 'primereact/tooltip';
import { useState } from 'react';
import { MenuItem } from 'primereact/menuitem';
import { Steps } from 'primereact/steps'
import { useRouter } from 'next/navigation';
import { Image } from 'primereact/image';

const Step5 = () => {
    const [resultImg] = useState('/demo/images/galleria/arya.png');

    const [activeIndex, setActiveIndex] = useState<number>(4);
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
    const HomeClick = async () => {
        router.push('/');
    };
    const BackClick = async () => {
        router.push('step4');
    };

    return (
        <div className="grid p-fluid input-demo">
            <div className="col-12">
                <div className="md:col-12">
                        <Steps model={items} activeIndex={activeIndex} onSelect={(e) => setActiveIndex(e.index)} />
                </div>
                <div className="card">
                    <div className="border-1 surface-border border-round m-1 text-center py-5">
                        <h5>Your result</h5>
                        <div className="flex justify-content-center">
                            <Image src={resultImg} width="400" />
                        </div>
                    </div>
                    <div className="flex justify-content-center gap-2">
                        <div className='car-buttons mt-5'>
                            <Button 
                                tooltip='Download'
                                tooltipOptions={{ position: 'top' }}
                                className='mr-2 p-button p-component p-button-icon-only p-button-rounded'
                                icon="pi pi-cloud-download" 
                                rounded aria-label="Filter" 
                            />
                            <Button 
                                tooltip='Save'
                                tooltipOptions={{ position: 'top' }}
                                className='mr-2 p-button p-component p-button-icon-only p-button-rounded'
                                icon="pi pi-check" 
                                rounded severity="success" 
                                aria-label="Search" 
                            />
                        </div>
                    </div>

                </div>
                <div className="flex justify-content-center gap-2">
                    <Tooltip target=".back" content="Back to previous step" position="top" />
                    <Button 
                        label="Back" 
                        className="back" 
                        style={{ margin: '0.25em 0.25em', width: '150px' }} 
                        onClick={BackClick}
                    />
                    <Tooltip target=".Finish" content="Back to main page" position="top" />
                    <Button 
                        label="Finish" 
                        className="Finish" 
                        severity="success"
                        style={{ margin: '0.25em 0.25em', width: '150px' }} 
                        onClick={HomeClick}
                    />
                </div>
            </div>
        </div>
    );
};

export default Step5;