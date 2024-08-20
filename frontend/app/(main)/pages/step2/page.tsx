'use client';
import { Button } from 'primereact/button';
import { Checkbox, CheckboxChangeEvent } from 'primereact/checkbox';
import { RadioButton } from 'primereact/radiobutton';
import { Tooltip } from 'primereact/tooltip';
import { useState } from 'react';
import { MenuItem } from 'primereact/menuitem';
import { Steps } from 'primereact/steps'
import { useRouter } from 'next/navigation';

const Step2 = () => {
    const [radioAge, setRadioAge] = useState<string | null>(null);
    const [radioGender, setRadioGender] = useState<string | null>(null);
    const [checkboxValue, setCheckboxValue] = useState<string[]>([]);

    const onCheckboxChange = (e: CheckboxChangeEvent) => {
        setCheckboxValue((prev) => (e.checked ? [...prev, e.value] : prev.filter((val) => val !== e.value)));
    };
    const [activeIndex, setActiveIndex] = useState<number>(1);
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
        router.push('step3');
    };
    const BackClick = async () => {
        router.push('step1');
    };

    return (
        <div className="grid p-fluid input-demo">
            <div className="col-12">
                <div className="md:col-12">
                        <Steps model={items} activeIndex={activeIndex} onSelect={(e) => setActiveIndex(e.index)} />
                </div>
                <div className="card">
                    <h5>Select your option</h5>
                    <h6>Gender</h6>
                    <div className="grid">
                        {['Male', 'Female'].map((gender) => (
                            <div key={gender} className="col-12 md:col-3">
                                <div className="field-radiobutton">
                                    <RadioButton inputId={`gender-${gender}`} name="gender" value={gender} checked={radioGender === gender} onChange={(e) => setRadioGender(e.value)} />
                                    <label htmlFor={`gender-${gender}`}>{gender}</label>
                                </div>
                            </div>
                        ))}
                    </div>

                    <h6>Age</h6>
                    <div className="grid">
                        {['15 - 20', '21 - 40', '41 - 60'].map((ageRange) => (
                            <div key={ageRange} className="col-12 md:col-3">
                                <div className="field-radiobutton">
                                    <RadioButton inputId={`age-${ageRange}`} name="age" value={ageRange} checked={radioAge === ageRange} onChange={(e) => setRadioAge(e.value)} />
                                    <label htmlFor={`age-${ageRange}`}>{ageRange}</label>
                                </div>
                            </div>
                        ))}
                    </div>

                    <h6>Accessories</h6>
                    <div className="grid">
                        {['Hat', 'Glasses'].map((accessory) => (
                            <div key={accessory} className="col-12 md:col-4">
                                <div className="field-checkbox">
                                    <Checkbox inputId={`accessory-${accessory}`} name="accessory" value={accessory} checked={checkboxValue.includes(accessory)} onChange={onCheckboxChange} />
                                    <label htmlFor={`accessory-${accessory}`}>{accessory}</label>
                                </div>
                            </div>
                        ))}
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

export default Step2;